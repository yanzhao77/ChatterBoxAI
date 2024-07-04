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
        element_of_weather = "大雨倾盆"  # 自己想办法获得指定城市当日天气，固定字符串也行啊。#
        return element_of_weather
    except Exception as err:
        print(err.__str__())
        return ''
    finally:
        pass
        # browser.quit()


if __name__ == "__main__":
    tools = [
        Tool(
            name="get_weather_info",
            func=get_weather_info.run,
            description='''当您只需要有关天气的信息来回答问题时很有用,您应该输入不带引号的城市名称.'''
        )
    ]
    # GPT4ALL_MODEL = os.getenv('GPT4ALL_MODEL')
    # llm = GPT4All(model=GPT4ALL_MODEL,temp=0.1,device="gpu")
    # llm = Ollama(model="llama3",temperature=0.1)
    # llm = ChatOpenAI(temperature=0.1)
    # llm = BaichuanLLM(model='Baichuan4', temperature=0.1)
    llm = Tongyi(temperature=0.1)

    print(llm.__class__)

    agent = initialize_agent(
        llm=llm,
        tools=tools,
        verbose=True,
    )

    chain = agent  # 虽然没有|任何其它东西，但没有这一句，那么用百川LLM时会出错。

    msg1 = "口袋妖怪中的伊布有多少种形态?"
    print(f">>>{msg1}")
    chain.invoke(msg1)

    msg2 = "北京（中国城市）的天气怎么样"
    print(f">>>{msg2}")
    chain.invoke(msg2)
