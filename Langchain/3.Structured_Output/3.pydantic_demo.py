from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    
    name: str
    department: str  = "Computer"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt= 4, default= None, description="A floating value representing the cumulative frequency of the student")
    
# student1 = {"name": "RAM"}
student1 = {"name": "Rama", "age": "24", "email": "test@gmail.com", "cgpa":3.50}

student = Student(**student1)

print(type(student))
print(student.department)
print(student)


#### conversion of pydtantic to dict

student_dict = dict(student)
print(type(student_dict))
print(student_dict["name"])


## json  

student_json = student.model_dump_json()
print(student_json)