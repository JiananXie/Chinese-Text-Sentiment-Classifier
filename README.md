# Chinese-Text-Sentiment-Classifier
This repo records my first assignment on the course Introduction to Chinese Information Processing(CIP)

Due to file size limitations, the datasets used to train the model are not uploaded to the warehouse.

You can download the dataset used here through:
  - (Weibo)https://www.biendata.xyz/ccf_tcci2018/datasets/emotion/ (used to train model3.pth and model6.pth)
  - (Sina)https://www.biendata.xyz/ccf_tcci2018/datasets/tcci_tag/19 (used to train model3.pth and model6.pth)
  - (ChnSentiCorp_htl_all)https://github.com/SophonPlus/ChineseNlpCorpus (used to train model2.pth)

### Model
The main model we used is BERT which is pretrained by HFL, you can load this pretrained model in the path of 'hfl/chinese-bert-wwm-ext'. We use this model to do fine-tuning on datasets consisting of text labeled among null, like, sadness, disgust, anger, happiness. Then we save the weights in the form of 'modelx.pth'(x can be 2,3,6). model2.pth means these weights are trained on a two-labeled dataset containing positive and negative. model3.pth means these weights are trained on a three-labeled dataset containing positive, null and nagative. model6.pth means these weights are trained on a six-labeled dataset containing null, like, sadness, disgust, anger and happiness. Of course, you can train the model on your emotion classification dataset and save your own models.

### Dataset
ChnSentiCorp_htl_all.csv is derived from https://github.com/SophonPlus/ChineseNlpCorpus, no more processing is operated on except for deleting a nan value. Positive_null_negative.csv is derived by integrating like, happiness into positive and integrating sadness,disgust,anger into negative on Sina and Weibo dataset, then combine the new 3-labeled dataset with ChnSentiCorp_htl_all to get the final training set. And to do 6-labeled emotion classification task, we use the dataset obtained by combine the Sina and Weibo dataset.

### Training
run `python BertSentiment.py`  in the terminal and wait for 1-2 hours(maybe longer which depends on your dataset).

### Result
We can do an excellent 2-labeled emotion classification with 92.5% acc. And we achieve 85.9% acc in 3-labeled emotion classification task. In 6-labeled task, we can only achieve 61.5%.

### Display
Here we complete a simple webpage to display our model, you can type into your words and it will return the emotion contained between words on the page. The emotion label displayed depends on what kind of the dataset you feed it. So, you can realize your own Chinese-Text-Sentiment-Classifier.

To run this web page, you only need to run `python app.py`, then open the '情感分析.html' file.

### In the end
It's just a very simple project which serves as the first homework of a humanities course. So, the model needs large amount of improvements to achieve better performance. Welcome to train your own model based on mine.
