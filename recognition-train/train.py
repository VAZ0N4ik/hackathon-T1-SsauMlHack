import datasets
import evaluate
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer, AutoTokenizer, AutoModel
from PIL import Image 
import os
from os import listdir



df = pd.read_csv('train/.tsv', sep = '\t')
df.columns = ['text', 'label']



train, test = train_test_split(df, test_size=0.3)
train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")



img = Image.open('train/aa1.png').convert("RGB")




def ocr_image(src_img):
  pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]



folder_dir = './train'
for images in os.listdir(folder_dir):
	if (images.endswith(".png")):
		ocr_image()




tokenized_train = train
tokenized_test = test

metric = evaluate.load('f1')
def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
	output_dir = 'test_trainer_log',
	evaluation_strategy = 'epoch',
	per_device_train_batch_size = 6,
	per_device_eval_batch_size = 6,
	num_train_epochs = 5,
	report_to='none')

trainer = Trainer(
	model = model,
	args = training_args,
	train_dataset = tokenized_train,
	eval_dataset = tokenized_test,
	compute_metrics = compute_metrics)

trainer.train()


save_directory = './pt_save_pretrained'
model.save_pretrained(save_directory)