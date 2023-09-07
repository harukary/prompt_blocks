import json
from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.output_parsers import PydanticOutputParser

class PydanticOutputParserJa(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()
        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema,ensure_ascii=False)
        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)