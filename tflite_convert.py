import tensorflow as tf


def convert_model_to_tflite(model_path, lite_model_path, reduce_model_size):
	converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_path)
	if reduce_model_size:
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	
	tflite_buffer = converter.convert()
	open(lite_model_path, 'wb').write(tflite_buffer)

	print('TFLite model created.')


if __name__ == '__main__':
	model_path = 'bert_qa_pt.tflite'
	convert_model_to_tflite('/home/luis_ernesto951008/bert/saved_model/1589825625', model_path, True)
