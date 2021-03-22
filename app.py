import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from fbprophet import Prophet

# time seiries 야후 금융에서 주식정보를 제공하는 라이브러리: y-finance 이용
# 주식정보를 불러오고 차트 그리기

# 해당 주식에 대한 트위터 글을 불러 올 수 있는 API 사용
# stocktwits.com에서 제공하는 Restful API 호출 및 데이터 가져오기

def main():
    st.header('Online Stock Price Ticker')
    
    # yfinance 실행 pip install yfinance
    symbol = st.text_input('심볼 입력: ')
    data = yf.Ticker(symbol)
    today = datetime.now().date().isoformat()
    # 오늘 날짜로 최신화().날짜부분만().문자열로()
 
    df = data.history(start='2010-06-01', end=today)

    st.dataframe(df)

    st.subheader('종가')
    st.line_chart(df['Close'])
    
    st.subheader('거래량')
    st.line_chart(df['Volume'])

    # yfinance 라이브러리만의 정보
    # data.info
    # data.calendar
    # data.major_holders
    # data.institutional_holders
    # data.recommendations
    div_df = data.dividends
    st.dataframe(div_df.resample('Y').sum())

    new_df = div_df.reset_index()
    new_df['Year'] = new_df['Date'].dt.year

    st.dataframe(new_df)

    fig = plt.figure()
    plt.bar(new_df['Year'], new_df['Dividends'])
    st.pyplot(fig)

    # 여러 주식 데이터를 한번에 보여주기
    favorites = ['msft', 'tsla', 'nvda', 'aapl', 'amzn']
    f_df = pd.DataFrame()
    for stock in favorites:
        f_df[stock] = yf.Ticker(stock).history(start = '2010-01-01', end = today)['Close']
    
    st.dataframe(f_df)
    st.line_chart(f_df)


    #---------------------------------


    # # pip install requests
    # # stocktwits의 API 호출
    # res = requests.get('https://api.stocktwits.com/api/2/streams/symbol/{}.json'.format(symbol))

    # # JSON 형식이라서 .json() 이용
    # res_data = res.json()

    # # 파이썬의 딕셔너리와 리스트의 조합으로 사용 가능
    # st.write(res_data)

    # for message in res_data['messages']:

    #     col1, col2 = st.beta_columns([1,4])
        
    #     with col1:
    #         st.image(message['user']['avatar_url'])
    #         st.write(message['user']['username'])
        
    #     with col2 :   
    #         st.write(message['body'])
    #         st.markdown(message['created_at'])


    

    #---------------------------------


    
    # conda install -c conda-forge fbprophet
    # 프로펫 설치
    p_df = df.reset_index()
    p_df.rename(columns = {'Date':'ds','Close':'y'}, inplace = True)

    # st.dataframe(p_df)

    m = Prophet()
    m.fit(p_df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    st.dataframe(forecast)
    
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

if __name__ == '__main__':
    main()