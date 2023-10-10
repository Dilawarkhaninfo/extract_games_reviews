import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load your CSV file into a DataFrame
df = pd.read_csv('your_dataset.csv')

# Assuming your CSV has 'ID' and 'comment' columns
X = df['comment']

# Convert 'comment' column to strings
df['comment'] = df['comment'].astype(str)

# Perform sentiment analysis using VADER
sid = SentimentIntensityAnalyzer()
df['compound'] = df['comment'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Assign sentiments based on the compound score
df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

# Create a CountVectorizer
count_vectorizer = CountVectorizer()

# Convert the comments to a sparse matrix of token counts
X_counts = count_vectorizer.fit_transform(df['comment'])

# Train an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_counts, df['sentiment'])

# Now, let's count the sentiments for each game ID
game_ids = df['ID'].unique()

for game_id in game_ids:
    game_comments = df[df['ID'] == game_id]['comment']
    game_comments_counts = count_vectorizer.transform(game_comments)
    predictions = svm_model.predict(game_comments_counts)

    # Counting the sentiments
    negative_count = sum(predictions == 'negative')
    positive_count = sum(predictions == 'positive')
    neutral_count = sum(predictions == 'neutral')

    print(f"Game ID: {game_id} - Negative: {negative_count}, Positive: {positive_count}, Neutral: {neutral_count}")
