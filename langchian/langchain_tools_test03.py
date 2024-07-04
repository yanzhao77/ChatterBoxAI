import os
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.tools import tool
from langchain.agents import AgentType
from langchain import LLMMathChain
from langchain_community.llms.tongyi import Tongyi

api_key = "sk-3f2d7473809f4f0492976b33f3146299"
os.environ["DASHSCOPE_API_KEY"] = api_key


# 简单定义函数作为一个工具

def personal_info(name: str) -> str:
    info_list = {
        "Artorias": {
            "name": "Artorias",
            "age": 18,
            "sex": "Male",
        },
        "Furina": {
            "name": "Furina",
            "age": 16,
            "sex": "Female",
        },
    }
    try:
        if name not in info_list:
            return ""
        return info_list[name]
    except Exception as err:
        print(err.__str__())
        return ''
    finally:
        pass


llm = Tongyi(temperature=0.1)

# 自定义工具字典
tools = (
    # 这个就是上面的llm-math工具
    Tool(
        name="Calculator",
        description="当你需要回答数学问题时很有用.",
        func=LLMMathChain.from_llm(llm=llm).run,
        coroutine=LLMMathChain.from_llm(llm=llm).arun,
    ),
    # 自定义的信息查询工具，声明要接收用户名字，并会给出用户信息
    Tool(
        name="personal_info",
        func=personal_info,
        description="当你需要回答有关某人的问题时很有用，输入人名，然后你将获得姓名和年龄信息."
    )
)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 提问，询问Furina用户的年龄的0.43次方
rs = agent.run("Furina 的年龄的 0.43 次方是多少?")
print(rs)
