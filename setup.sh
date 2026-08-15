pip install -r requirements.txt

mkdir models
wget https://huggingface.co/ragavsachdeva/magiv2/resolve/main/pytorch_model.bin -O ./models/pytorch_model.bin
python -m scripts.filter_model_weights

echo "Setup complete"