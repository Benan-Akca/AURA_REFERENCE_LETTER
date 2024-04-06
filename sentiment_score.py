from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(MODEL)

def tokenize_sentences(text):
    sentence_list = sent_tokenize(text)
    return sentence_list

def get_sentiment_score(text):
    #text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    if encoded_input["input_ids"].shape[1]>512:
        encoded_input["input_ids"] = encoded_input["input_ids"][:,0:512]
        encoded_input["attention_mask"] = encoded_input["attention_mask"][:,0:512]
    output = model_sentiment(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores[2]

def get_sentiment_df(df, sentence_column):
    scores_list = [0]*df[sentence_column].shape[0]
    for inx, letter in tqdm(enumerate(df[sentence_column].values)):
        score = 0
        sentences = tokenize_sentences(letter)
        for sentence in sentences:
            score += get_sentiment_score(sentence)
        scores_list[inx] = score/len(sentences)

    df["sentiment"] = scores_list
        
    return df
