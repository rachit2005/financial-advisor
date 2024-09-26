from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
from textblob import TextBlob
import yfinance as yf
import datetime
from googlesearch import search
import io
import base64

app = Flask(__name__)

# Global variables
tickers = []
tracker = None

# Initialize model and other global variables
model = None
loss_function = None
optimizer = None

def google_search(query):
    search_results = search(query, num_results=10)
    return search_results

# Convert the data into a supervised learning problem
def create_dataset(dataset, look_back=60):
    X, Y = [], [] # creating an empty list
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def portfolio_return(weights, returns_tensor):
    return torch.sum(weights * torch.mean(returns_tensor, dim=0))

def portfolio_volatility(weights, returns_tensor):
    cov_matrix = torch.cov(returns_tensor.T)
    return torch.sqrt(torch.matmul(weights.T, torch.matmul(cov_matrix, weights)))

def sharpe_ratio(weights, returns_tensor, risk_free_rate=0.01):
    return (portfolio_return(weights, returns_tensor) - risk_free_rate) / portfolio_volatility(weights, returns_tensor)

def negative_sharpe_ratio(weights, returns_tensor, risk_free_rate=0.01):
    return -sharpe_ratio(weights, returns_tensor, risk_free_rate)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Initialize model, loss function, and optimizer once
def initialize_model():
    global model, loss_function, optimizer
    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

initialize_model()

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import datetime

app = Flask(__name__)

# Global variable for expense tracker
tracker = None

class ExpenseTracker:
    def __init__(self):
        self.expenses = pd.DataFrame(columns=['Date', 'Category', 'Amount', 'Description'])

    def add_expense(self, category, amount, description):
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        new_expense = pd.DataFrame([[date, category, float(amount), description]], columns=self.expenses.columns)
        self.expenses = pd.concat([self.expenses, new_expense], ignore_index=True)
        print(f"Added expense: {new_expense}")

    def get_expense_report(self):
        return self.expenses.groupby('Category').sum()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/track_expenses', methods=['GET', 'POST'])
def track_expenses():
    global tracker
    if tracker is None:
        tracker = ExpenseTracker()

    if request.method == 'POST':
        category = request.form['category']
        amount = request.form['amount']
        description = request.form['description']
        print(f"Received expense: Category={category}, Amount={amount}, Description={description}")
        tracker.add_expense(category, amount, description)
        return redirect(url_for('track_expenses'))

    report = tracker.get_expense_report() if tracker else pd.DataFrame()
    print(f"Expense Report: {report}")
    return render_template('track_expenses.html', report=report)


@app.route('/predict_stock', methods=['GET', 'POST'])
def predict_stock():
    global model, loss_function, optimizer
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        data = yf.download(stock_name, start="2010-01-01", end=str(datetime.datetime.today()).split()[0])
        data = data[['Close']]
        
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data) # normalising the data
        X, y = create_dataset(scaled_data)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.19917, random_state=42, shuffle=True)
        x_train, x_test, y_train, y_test = torch.from_numpy(x_train).type(torch.Tensor), torch.from_numpy(x_test).type(torch.Tensor), torch.from_numpy(y_train).type(torch.Tensor), torch.from_numpy(y_test).type(torch.Tensor)

        print(f'len of x train {len(x_train)}')
        print(f'len of x test {len(x_test)}')

        for epoch in range(10):
            for price, label in zip(x_train, y_train):
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))
                y_pred = model(price)
                loss = loss_function(y_pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch%5 == 0:
                print(f'epoch {epoch}')

        model.eval()
        test_predictions = []
        for i in range(len(x_test)):
            seq = x_test[i]
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))
                test_pred = model(seq)
                test_predictions.append(test_pred.item())

        test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].plot(data.index[-len(y_test_actual):], y_test_actual, label='Actual Price', color='black')
        ax[0].set_title(f'Actual stock price for {stock_name}')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Price')

        ax[1].plot(data.index[-len(test_predictions):], test_predictions, label='Predicted Price', color='red')
        ax[1].set_title(f'Stock Price Prediction for {stock_name}')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Price')
        ax[1].legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        img.close()

        return render_template('predict_stock.html', plot_url=plot_url)

    return render_template('predict_stock.html')

import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

@app.route('/optimise_portfolio', methods=['GET', 'POST'])
def optimise_portfolio():
    global tickers
    try:
        if request.method == 'POST':
            tickers_input = request.form.get('tickers')
            tickers = [ticker.strip() for ticker in tickers_input.split(',')]
            num_stocks = len(tickers)

            print(tickers)

            if num_stocks == 0:
                return render_template('optimise_portfolio.html', error="Please select at least one ticker.")

            try:
                port_data = yf.download(tickers, start="2010-01-01", end=str(datetime.datetime.today()).split()[0])['Adj Close']
                # print(port_data)
                if port_data.empty:
                    raise ValueError("No data found for the provided tickers.")
            except Exception as e:
                return render_template('optimise_portfolio.html', error=f"Failed to download data: {str(e)}")

            returns = port_data.pct_change().dropna()
            returns_tensor = torch.tensor(returns.values, dtype=torch.float32)

            # Initialize weights and ensure they sum to 1
            weights = Variable(torch.ones(num_stocks, requires_grad=True, dtype=torch.float32) / num_stocks , requires_grad=True)

            optimizer = torch.optim.SGD([weights], lr=0.01)
            print('know optimizer')
            num_epochs = 10
            for epoch in range(num_epochs):
                print('inside train loop')
                optimizer.zero_grad()

                # Create new weights to avoid in-place operations
                normalized_weights = torch.abs(weights)
                normalized_weights = normalized_weights / torch.sum(normalized_weights)

                loss = negative_sharpe_ratio(normalized_weights, returns_tensor)
                loss.backward()
                optimizer.step()

            optimal_return = portfolio_return(normalized_weights, returns_tensor).item()
            optimal_volatility = portfolio_volatility(normalized_weights, returns_tensor).item()
            optimal_sharpe_ratio = sharpe_ratio(normalized_weights, returns_tensor).item()

            # Plotting the portfolio weights
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(tickers, normalized_weights.detach().numpy())
            ax.set_title('Optimal Portfolio Allocation')
            ax.set_xlabel('Assets')
            ax.set_ylabel('Weight')

            # Save the plot to a buffer
            img = io.BytesIO()
            FigureCanvas(fig).print_png(img)
            img.seek(0)

            # Encode the image to base64
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            img.close()

            return render_template('optimise_portfolio.html', plot_url=plot_url, optimal_return=optimal_return, optimal_volatility=optimal_volatility, optimal_sharpe_ratio=optimal_sharpe_ratio)

    except Exception as e:
        return render_template('optimise_portfolio.html', error=str(e))

    return render_template('optimise_portfolio.html')


@app.route('/search_query', methods=['GET', 'POST'])
def search_query():
    if request.method == 'POST':
        query = request.form['query']
        search_results = google_search(query)
        return render_template('search_query.html', search_results=search_results)

    return render_template('search_query.html')

if __name__ == '__main__':
    app.run(debug=True)
