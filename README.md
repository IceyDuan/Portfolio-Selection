# Portfolio-Selection
This is a fundamental project about Machine Learning and Qualitative Analysis by Python. The theme is Application of LSTM Neural Network in Optimizing-then-Predicting Winning and Losing Stocks.
The detailed process is in the paper uploaded in the folder of LaTex, you also could see the LaTex files and figures.
## Part I Markowitz Model Construction
The Markowitz mean-variance portfolio optimization model is designed to weigh the expected return and risk when investing. Investors with different levels of risk aversion have the
option of choosing different portfolios according to their objectives, thus constructing an optimal portfolio with efficient frontiers that represent all possible risk tolerances.

The raw data is from Tushare Website. The randomly selected 50 stocks has daily, weekly and monthly quotes of these stocks. Their efficient frontiers are constructed like below.

<img src="https://github.com/user-attachments/assets/e60048bd-9f20-40ea-80d6-81ce0bdad2ac" style="width: 40%; height: 40%;" align= center>
<img src="https://github.com/user-attachments/assets/60599a15-f0fb-4a9a-9291-1d0008430836" style="width: 40%; height: 40%;" align= center>

## Part II Dataset
We need to make some preparation for supervised learning. The stocks whose weight is infinity approaches zero can be seen as not included into the portfolio while others are included. The efficient frontiers are divided into 50 points. The result is like: <img src="https://github.com/user-attachments/assets/d95ccdb5-53f8-4751-a349-0ebbfb8a0085" style="float;">

## Part III LSTM machine learning
This part is about the neural network used to predict whether a stock has potential to be includede into your own portfolio, consider the stability and profits. This model is trained and tested by the previous dataset.

<img src="https://github.com/user-attachments/assets/1c88ad6a-14ea-49e6-837e-45b9ad2fbe9f" style="width: 40%; height: 40%;">
<img src="https://github.com/user-attachments/assets/abc58b8e-cd20-4084-8971-2a8699dc8cbe" style="width: 16%; height: 16%;">
