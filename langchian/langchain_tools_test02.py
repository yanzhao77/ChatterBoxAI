import os

from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.tools import tool
from langchain_community.llms.tongyi import Tongyi

api_key = "sk-3f2d7473809f4f0492976b33f3146299"
os.environ["DASHSCOPE_API_KEY"] = api_key


@tool
def get_weather_info(city: str) -> str:
    """Returns the weather forcast of a city(in China) of today."""
    try:
        print(str)
        element_of_weather = "暴雨转晴"  # 自己想办法获得指定城市当日天气，固定字符串也行啊。#
        return element_of_weather
    except Exception as err:
        print(err.__str__())
        return ''
    finally:
        pass



if __name__ == "__main__":
    tools = [
        Tool(
            name="get_weather_info",
            func=get_weather_info.run,
            description='''当您只需要有关天气的信息来回答问题时很有用,您应该输入不带引号的城市名称.'''
        )
    ]
    llm = Tongyi(temperature=0.1)
    agent = initialize_agent(
        llm=llm,
        tools=tools,
        verbose=True,
    )

    msg1 = "西安天气怎么样?"
    print(f">>>{msg1}")
    agent.invoke(msg1)

