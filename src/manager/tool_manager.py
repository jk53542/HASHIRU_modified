import importlib
import importlib.util
import os
from typing import List, Type, Any
import pip
from google.genai import types

from src.manager.budget_manager import BudgetManager
from src.manager.utils.singleton import singleton
from src.manager.utils.suppress_outputs import suppress_output
from src.tools.default_tools.tool_deletor import ToolDeletor
from src.manager.utils.streamlit_interface import output_assistant_response

toolsImported = []

TOOLS_DIRECTORIES = [os.path.abspath("./src/tools/default_tools"), os.path.abspath("./src/tools/user_tools")]

installed_packages = set()


def _resolve_tool_class_from_module(foo: Any, module_name: str) -> Type:
    """
    Resolve the Tool class for a module under default_tools/ or user_tools/.

    Standard tools define __all__ = ['ClassName']. ToolCreator output often omits __all__;
    then we pick a public class with inputSchema + run().
    """
    if hasattr(foo, "__all__") and getattr(foo, "__all__", None):
        primary = foo.__all__[0]
        if isinstance(primary, str) and hasattr(foo, primary):
            cls = getattr(foo, primary)
            if isinstance(cls, type):
                return cls
    candidates: list[tuple[str, type]] = []
    for name in dir(foo):
        if name.startswith("_"):
            continue
        obj = getattr(foo, name, None)
        if not isinstance(obj, type):
            continue
        if getattr(obj, "inputSchema", None) is None:
            continue
        if not callable(getattr(obj, "run", None)):
            continue
        candidates.append((name, obj))
    if not candidates:
        raise AttributeError(
            "no __all__ and no class with inputSchema + run() "
            "(see src/tools/user_tools/WordCountExtractor.py)"
        )
    mn = module_name.lower().replace("_", "")
    for name, obj in candidates:
        n = name.lower().replace("_", "")
        if n == mn or mn in n or n in mn:
            return obj
    return candidates[0][1]


class Tool:
    def __init__(self, toolClass):
        suppress_output(self.load_tool)(toolClass)
        
    def load_tool(self, toolClass):
        self.tool = toolClass()
        self.inputSchema = self.tool.inputSchema
        self.name = self.inputSchema["name"]
        self.description = self.inputSchema["description"]
        self.dependencies = self.tool.dependencies
        self.create_resource_cost = self.inputSchema.get("create_resource_cost", 0)
        self.invoke_resource_cost = self.inputSchema.get("invoke_resource_cost", 0)
        self.create_expense_cost = self.inputSchema.get("create_expense_cost", 0)
        self.invoke_expense_cost = self.inputSchema.get("invoke_expense_cost", 0)
        if self.dependencies:
            self.install_dependencies()
    
    def install_dependencies(self):
        for package in self.dependencies:
            if package in installed_packages:
                continue
            try:
                __import__(package.split('==')[0])
            except ImportError:
                print(f"Installing {package}")
                if '==' in package:
                    package = package.split('==')[0]
                pip.main(['install', package])
            installed_packages.add(package)

    def run(self, query):
        return self.tool.run(**query)

@singleton
class ToolManager:
    toolsImported: List[Tool] = []
    budget_manager: BudgetManager = BudgetManager()
    is_creation_enabled: bool = True
    is_invocation_enabled: bool = True

    def __init__(self):
        self.load_tools()
        self._output_budgets()
    
    def set_creation_mode(self, status: bool):
        self.is_creation_enabled = status
        if status:
            output_assistant_response("Tool creation mode is enabled.")
        else:
            output_assistant_response("Tool creation mode is disabled.")
    
    def set_invocation_mode(self, status: bool):
        self.is_invocation_enabled = status
        if status:
            output_assistant_response("Tool invocation mode is enabled.")
        else:
            output_assistant_response("Tool invocation mode is disabled.")
    
    def _output_budgets(self):
        output_assistant_response(f"Resource budget Remaining: {self.budget_manager.get_current_remaining_resource_budget()}")
        output_assistant_response(f"Expense budget Remaining: {self.budget_manager.get_current_remaining_expense_budget()}")

    def load_tools(self):
        newToolsImported = []
        for directory in TOOLS_DIRECTORIES:
            for filename in os.listdir(directory):
                if filename.endswith(".py") and filename != "__init__.py":
                    module_name = filename[:-3]
                    path = os.path.join(directory, filename)
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, path)
                        if spec is None or spec.loader is None:
                            print(f"Skipping tool {module_name!r}: could not load spec")
                            continue
                        foo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(foo)
                        toolClass = _resolve_tool_class_from_module(foo, module_name)
                        toolObj = Tool(toolClass)
                        newToolsImported.append(toolObj)
                        if toolObj.create_resource_cost is not None:
                            self.budget_manager.add_to_resource_budget(toolObj.create_resource_cost)
                        if toolObj.create_expense_cost is not None:
                            self.budget_manager.add_to_resource_budget(toolObj.create_expense_cost)
                    except Exception as e:
                        print(f"Skipping invalid tool module {module_name!r} ({path}): {e}")
        self.toolsImported = newToolsImported

    def runTool(self, toolName, query):
        if not self.is_invocation_enabled:
            raise Exception("Tool invocation mode is disabled")
        if toolName == "ToolCreator":
            if not self.is_creation_enabled:
                raise Exception("Tool creation mode is disabled")
        self._output_budgets()
        for tool in self.toolsImported:
            if tool.name == toolName:
                if tool.invoke_resource_cost is not None:
                    if not self.budget_manager.can_spend_resource(tool.invoke_resource_cost):
                        raise Exception("No resource budget remaining")
                if tool.invoke_expense_cost is not None:
                    self.budget_manager.add_to_resource_budget(tool.invoke_expense_cost)
                return tool.run(query)
        self._output_budgets()
        return {
            "status": "error",
            "message": f"Tool {toolName} not found",
            "output": None
        }
    

    def getTools(self):
        # When tool invocation is off, do not expose declarations to the model — otherwise it
        # still emits function_call parts and every call fails in runTool(), causing loops.
        if not self.is_invocation_enabled:
            return []
        toolsList = []
        for tool in self.toolsImported:
            parameters = types.Schema()
            parameters.type = tool.inputSchema["parameters"]["type"]
            properties = {}
            for prop, value in tool.inputSchema["parameters"]["properties"].items():
                properties[prop] = types.Schema(
                    type=value["type"],
                    description=value["description"]
                )
            parameters.properties = properties
            parameters.required = tool.inputSchema["parameters"].get("required", [])
            function = types.FunctionDeclaration(
                name=tool.inputSchema["name"],
                description=tool.inputSchema["description"],
                parameters=parameters,
            )
            toolType = types.Tool(function_declarations=[function])
            toolsList.append(toolType)
        # @@ DEBUGGING CALL
        # print(toolsList)
        return toolsList
    
    def delete_tool(self, toolName, toolFile):
        try:
            tool_deletor = ToolDeletor()
            tool_deletor.run(name=toolName, file_path=toolFile)
            for tool in self.toolsImported:
                if tool.name == toolName:
                    # remove budget for the tool
                    if tool.create_resource_cost is not None:
                        self.budget_manager.remove_from_resource_expense(tool.create_resource_cost)
                    if tool.create_expense_cost is not None:
                        self.budget_manager.remove_from_resource_expense(tool.create_expense_cost)
                    self.toolsImported.remove(tool)
                    return {
                        "status": "success",
                        "message": f"Tool {toolName} deleted",
                        "output": None
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tool {toolName} not found",
                "output": None
            }