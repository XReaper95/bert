import json
from zipfile import ZipFile
from pathlib import Path
from typing import Union

import tensorflow as tf


def get_model_name(meta_path: Union[str, Path]) -> str:
	meta_path = Path(meta_path)
	if meta_path.exists():
		with meta_path.open() as f:
			metadata = json.load(f)
			seed = metadata["seed"]
			epochs = metadata["epochs"]
			batch = metadata["batch"]

			return f"android_model_{seed}_E{epochs}B{batch}.tflite"
	else:
		print(f"Metadata does not exists at {meta_path}")


def delete_old_model(model_path: Union[str, Path]):
	model_path = Path(model_path)
	if model_path.exists():
		model_path.unlink()
		print("Old model deleted!")


def compress_model(model_name, model_path, metadata_path):
	file_name = f"../compress/{model_name.split('.')[0]}.zip"

	zip_obj = ZipFile(file_name, 'w')
	zip_obj.write(model_path)
	zip_obj.write(metadata_path)
	zip_obj.close()

	print(f'File compressed as {file_name}')


def convert_model_to_tflite(model_path, lite_model_path, reduce_model_size):
	converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_path)
	# if reduce_model_size:
	# 	converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
	# 	converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	
	tflite_buffer = converter.convert()
	open(lite_model_path, 'wb').write(tflite_buffer)

	print('TFLite model created.')


if __name__ == '__main__':
	# metadata_path = '../android/meta.json'
	# model_name = get_model_name(metadata_path)
	model_path = 'bert_qa_pt.tflite'
	# delete_old_model(model_path)
	convert_model_to_tflite('/home/luis_ernesto951008/bert/saved_model/1589825625', model_path, True)
	# compress_model(model_name, model_path, metadata_path)
