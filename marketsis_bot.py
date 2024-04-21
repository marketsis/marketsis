import telebot
import json
import os
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, mean_squared_error
from prophet import Prophet
from math import sqrt
from telebot import types
import matplotlib


matplotlib.use('Agg')

TOKEN = ''

bot = telebot.TeleBot(TOKEN)

forecasting_final = None

@bot.message_handler(commands=['start'])
def start(message):
    keyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    briefBtn = types.KeyboardButton(text="📖 О инструменте")
    trendTickersBtn = types.KeyboardButton(text="📈 Трендовые тикеры")
    keyboard.add(briefBtn, trendTickersBtn)
    user = message.from_user.first_name     
    
    bot.send_message(chat_id=message.from_user.id, text= f"Приветствую, {user}. Глобальный предиктор готов к работе!", reply_markup=keyboard)


@bot.message_handler(func=lambda message: message.text == "📖 О инструменте")
def sendBrief(message):
    bot.send_message(message.chat.id, '''marketsis - это инструмент, который предоставляет актуальную информацию о акциях и их динамике движения, последние новости и данные о институциональных держателях определенных акций. С помощью этого бота вы сможете отслеживать тенденции котировок акций, анализировать графики и прогнозировать временные ряды - это поможет вам принимать более обоснованные инвестиционные решения.

Чтобы начать использовать бота, просто отправьте соответствующий запрос c идентификатором акции (тикером), выберете прогнозный период, и бот отправит вам информацию в виде достроеного графика котировок акции и графики корреляции.
Расширенная информация по запрашиваемой компании предоставляется посредством нажатия на пункты в клавиатурном меню сообщени - Вы можете запросить актуальные списки релевантных новостей и институциональных держателей акций с отображением подробных данных.
После запроса графика предоставляется возможность вывести характеристики качества модели. Показатель "Средняя абсолютная ошибка в процентах" (MAPE) отображает средний процент ошибки прредсказания модели в сравнении с актуальными данными в ходе ее обучения. Благодаря этим показателями Вы можете удостовериться в качестве обучения модели предсказания.

Не является финансовой рекомендацией.''')


@bot.message_handler(func=lambda message: message.text == "📈 Трендовые тикеры")
def sendTrendTickers(message):
    bot.send_message(message.chat.id, text=f"Здесь Вы можете просмотреть актуальные трендовые тикеры акций:\n\nhttps://finance.yahoo.com/lookup")


def modelQualityAssessment(forecasting_final):
    MAE = mean_absolute_error(forecasting_final['yhat'],forecasting_final['y'])
    print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))

    MEDAE = median_absolute_error(forecasting_final['yhat'],forecasting_final['y'])
    print('Median Absolute Error (MedAE): ' + str(np.round(MEDAE, 2)))

    MSE = mean_squared_error(forecasting_final['yhat'],forecasting_final['y'])
    print('Mean Squared Error (MSE): ' + str(np.round(MSE, 2)))

    RMSE = sqrt(int(mean_squared_error(forecasting_final['yhat'],forecasting_final['y'])))
    print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE, 2)))

    MAPE = mean_absolute_percentage_error(forecasting_final['yhat'],forecasting_final['y'])
    print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')
    
    qualityAssessmentMessage = f'''Mean Absolute Error (MAE): {str(np.round(MAE, 2))}
Median Absolute Error (MedAE): {str(np.round(MEDAE, 2))}
Mean Squared Error (MSE): {str(np.round(MSE, 2))}
Root Mean Squared Error (RMSE): {str(np.round(RMSE, 2))}
Mean Absolute Percentage Error (MAPE): {str(np.round(MAPE, 2))} %
    '''
    return qualityAssessmentMessage

def generateCharts(hist, forecast, ticker, model):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(hist.ds, hist.y, label='Текущие данные') 
    ax.plot(forecast.ds, forecast.yhat, linestyle='--', label='Прогнозируемые данные')  
    ax.fill_between(forecast.ds, forecast['yhat_upper'], forecast['yhat_lower'], alpha=0.2)
    ax.legend(loc='upper left')
    ax.set_title(ticker)
    plt.savefig("plot.png")
    fig = model.plot_components(forecast)
    plt.savefig("plot_components.png")

    plt.close()

def getPredictInfo(hist, forecast, ticker):
    entryDate = hist.iloc[-1]['ds'].date()
    entryPrice = round(hist.iloc[-1]['y'],2)
    targetDate = forecast.iloc[-1]['ds'].date()
    targetPrice = round(forecast.iloc[-1]['yhat'],2)
    profitForecast = round((targetPrice-entryPrice)/entryPrice*100, 2)
    profitForecastString = f"Прогнозируемая прибыль: {profitForecast} %" if profitForecast > 0 else f"Прогнозируемый убыток: {profitForecast} %"
    global forecasting_final
    forecasting_final = pd.merge(forecast, hist, how='inner', left_on = 'ds', right_on = 'ds')
    
    predictInfoMessage = f'''Тикер: {ticker}
Дата: {entryDate}
Входная цена: {entryPrice}
Прогнозная дата: {targetDate}
Целевая цена: {targetPrice}
{profitForecastString}'''
    return predictInfoMessage

def getHistoricalData(stock, actualPer):
    hist = stock.history(period=actualPer) 
    hist.reset_index(inplace=True)
    hist = hist[["Date", "Close"]]
    hist.rename(columns={"Date":"ds", "Close":"y"}, inplace=True)

    hist = hist.dropna(how='any')
    hist['ds'] = pd.to_datetime(hist['ds'])
    hist['ds'] = hist['ds'].dt.tz_localize(None)
    # hist1 = hist
    # hist = hist.query("ds <= '2022-09-06'")
    return hist

def genMarkup(ticker):
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton(f"📰 Cписок релевантных новостей {ticker}", callback_data=f"getRelevantNews {ticker}"),
               InlineKeyboardButton(f"🏦 Cписок институциональных держателей {ticker}", callback_data=f"getInstitutionalHolders {ticker}"),
               InlineKeyboardButton(f"⚠️ Характеристика качества модели", callback_data=f"getModelQualityAssessment"))
    return markup

def forecastPeriodMarkup(ticker):
    markup = InlineKeyboardMarkup()
    markup.row_width = 3
    markup.add(InlineKeyboardButton(f"6 мес.", callback_data=f"half-year {ticker}"),
            InlineKeyboardButton(f"1 год", callback_data=f"year {ticker}"),
            InlineKeyboardButton(f"2 года", callback_data=f"2years {ticker}"))
    return markup

stockItem = lambda t : yf.Ticker(t)

@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    try:
        if call.data.split(' ')[0] == "getRelevantNews":
            # ticker = call.data.split(' ')[1]
            # stock = yf.Ticker(ticker)
            
            news = stockItem(call.data.split(' ')[1]).get_news()
            bot.send_message(call.from_user.id, text=f"📰 Cписок релевантных новостей {call.data.split(' ')[1]}")
            for item in news:
                bot.send_message(call.from_user.id, text=f'''{str(item['title'])}
{str(item['link'])}

Дата публикации: {str(datetime.datetime.fromtimestamp(item['providerPublishTime']))}
Источник: {str(item['publisher'])}''')

            bot.answer_callback_query(call.id, f"Выведен список актуальных новостей, связанных с {call.data.split(' ')[1]}")
        elif call.data.split(' ')[0] == "getInstitutionalHolders":
            # ticker = call.data.split(' ')[1]
            # stock = yf.Ticker(ticker)
            holders = stockItem(call.data.split(' ')[1]).get_institutional_holders()
            bot.send_message(call.from_user.id, text=f"🏦 Cписок институциональных держателей: {call.data.split(' ')[1]}")
            for index, item in holders.iterrows():
                bot.send_message(call.from_user.id, text=f'''Держатель: {str(item['Holder'])}
Кол-во акций: {str(item['Shares'])}
Дата публикации информации: {str(item['Date Reported'])}
Процент от общей доли: {str(item['% Out'])}
Стоимость активов: {str(item['Value'])}''')
        elif call.data == "getModelQualityAssessment" and not forecasting_final.empty:
            bot.send_message(call.from_user.id, text=f"Параметры оценки качества отображают, насколько теоретические вычисления по построенной модели отклоняются от экспериментальных данных:\n\n{modelQualityAssessment(forecasting_final)}")
        elif call.data.split(' ')[0] == "half-year":
            constuctModel(stockItem(call.data.split(' ')[1]), call.data.split(' ')[1], call.from_user.id, "1y", 183)
        elif call.data.split(' ')[0] == "year":
            constuctModel(stockItem(call.data.split(' ')[1]), call.data.split(' ')[1], call.from_user.id, "2y", 365)
        elif call.data.split(' ')[0] == "2years":
            constuctModel(stockItem(call.data.split(' ')[1]), call.data.split(' ')[1], call.from_user.id, "4y", 730)
        else:
            bot.answer_callback_query(call.id, f"Опция невозможна для этой акции")
    except:
        bot.answer_callback_query(call.id, f"Опция невозможна для этой акции")

def constuctModel(stock, ticker, chatId, actualPer, futurePer):
    try:

        plotConponentsMessage = f'''Компоненты анализируемого набора данных:
- График тренда;
- Отображение корреляции котировок и праздников;
- Ежегодная сезонность;
- Ежеквартальная сезонность.'''

        hist = getHistoricalData(stock, actualPer)
        model = Prophet(
            yearly_seasonality=20, 
            changepoint_prior_scale=0.05,  
            seasonality_mode='multiplicative', 
            weekly_seasonality=False,
            daily_seasonality=False, 
            holidays_prior_scale=0.05, 
        )

        model.add_country_holidays(country_name='USA')
        model.add_seasonality(name='quarterly', period=91, fourier_order=6)
        model.fit(hist)

        bot.send_message(chatId, text=f"Подождите, идет анализ...")

        future = model.make_future_dataframe(periods=futurePer)
        forecast = model.predict(future)

        plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

        generateCharts(hist, forecast, ticker, model)

        bot.send_message(chat_id=chatId, text=getPredictInfo(hist, forecast, ticker))
        with open('plot.png', 'rb') as f:
            bot.send_photo(chatId, f)

        last_message_id = bot.send_message(chat_id=chatId, text=plotConponentsMessage).message_id
        with open('plot_components.png', 'rb') as f:
            bot.send_photo(chatId, f, reply_markup=genMarkup(ticker))
    
    except Exception as ex:
        print(ex)

@bot.message_handler()
def handle_message(message):

    try:
        if os.path.exists("plot.png"):
            os.remove("plot.png")
            os.remove("plot_components.png")
        else:
            print("file not found")
    except: 
        print("plt updated")

    try:
        ticker = str(message.text).strip().upper()
        stock = None

        tickerNotFoundMessage = f'''Анализ акций {ticker} невозможен.

- Возможно допущена ошибка при написании тикера?
- Акции {ticker} сняты с торгов?
- Отсутствуют исторические данные по {ticker}?
- {ticker} - это акции российской компании? Добавьте ".ME" к тикеру ({ticker}.ME)

Также проверьте идентификатор на сайте: 
https://finance.yahoo.com/'''

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
        except:
            bot.send_message(chat_id=message.from_user.id, text=tickerNotFoundMessage)
            return

        bot.send_message(message.from_user.id, text=f"Выберете прогнозируемый период", reply_markup=forecastPeriodMarkup(ticker))

    except Exception as ex:
        print(ex)

bot.infinity_polling()