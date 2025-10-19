
import pandas as pd
def load_dataset():
    churn_df = pd.read_csv("../data/raw/churn.csv" , encoding="ISO-8859-1")
    retail_df = pd.read_csv("../data/external/online_retail.csv" , encoding="ISO-8859-1")
    sentiment_df = pd.read_csv("../data/external/sentiment.csv", encoding="ISO-8859-1")
    logs_df = pd.read_csv("../data/external/user_behavior_dataset.csv", encoding="ISO-8859-1")
    
    return churn_df , retail_df , sentiment_df , logs_df

def clean_churn(churn_df):
    churn_df = churn_df.dropna(subset=["customerID"])
    churn_df["churn"] = churn_df["churn"].apply(lambda x: 1 if x == "Yes" else 0)
    churn_df["customerID"] = churn_df["customerID"].astype(int)
    churn_df["churn"] = churn_df["churn"].astype(int)
    return churn_df

def clean_retail(retail_df):
    retail_df = retail_df.dropna(subset=["customerID"])
    retail_df = retail_df[(retail_df["Quantity"] > 0) & (retail_df["UnitPrice"] > 0)]
    retail_df["InvoiceDate"] = pd.to_datetime(retail_df["InvoiceDate"])
    retail_df["customerID"] = retail_df["customerID"].astype(int)
    return retail_df
def clean_sentiment(sentiment_df):
    sentiment_df["sentiment_score"] = sentiment_df["sentiment_score"].fillna(0)
    sentiment_df["sentiment_label"] = sentiment_df["sentiment_label"].fillna("neutral")
    sentiment_df["customerID"] = sentiment_df["customerID"].astype(int)
    sentiment_df["sentiment_label"] = sentiment_df["sentiment_score"].apply(
    lambda s: 1 if s > 0.05 else (-1 if s < -0.05 else 0)
)
    return sentiment_df

    return sentiment_df
def clean_logs(logs_df):
    logs_df = logs_df.dropna(subset=["customerID"])
    logs_df['Age'] = logs_df['Age'].fillna(logs_df['Age'].median())
    logs_df['Gender'] = logs_df['Gender'].fillna('Unknown')
    logs_df['Device Model'] = logs_df['Device Model'].fillna('Unknown')
    logs_df['Operating System'] = logs_df['Operating System'].fillna('Unknown')
    logs_df['Gender'] = logs_df['Gender'].apply(lambda x: 1 if x == "Male" else 0)
    return logs_df

def merge_dataset(churn , retail , sentiment , logs):
    df = pd.merge(churn , retail , how="inner" , on="customerID")
    df = pd.merge(df , sentiment , how="inner" , on="customerID")
    df = pd.merge(df , logs , how="inner" , on="customerID")
    return df

def aggregate_retail(df):
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"] , errors ="coerce")
    agg = df.groupby("customerID").agg({
        'InvoiceNo' : 'nunique',
        'Quantity' : 'sum',
        'InvoiceDate' : 'max',
        'TotalPrice': ["sum","mean"]
    })
    agg.columns = ["NumPurchases" , "TotalQuantity" , "LatestPurchaseDate" , "TotalSpent" , "AvgOrderValue"]
    agg = agg.reset_index()
    current_time = df["InvoiceDate"].max()
    agg["RecencyDays"] = (current_time - agg["LatestPurchaseDate"]).dt.days 
    return agg

def feature_engineer(df):
    df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"]) 

def preprocess():
    churn_df , retail_df , sentiment_df , logs_df = load_dataset()
    churn_df = clean_churn(churn_df)
    retail_df = clean_retail(retail_df)
    retail_df = aggregate_retail(retail_df)
    sentiment_df = clean_sentiment(sentiment_df)
    logs_df = clean_logs(logs_df)
    final_df = merge_dataset(churn_df,retail_df,sentiment_df, logs_df)
    final_df = feature_engineer(final_df)
    return final_df 

    
    
