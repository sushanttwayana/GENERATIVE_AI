from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda