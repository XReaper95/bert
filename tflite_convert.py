import tensorflow as tf


def convert_model_to_tflite(model_path, lite_model_path, reduce_model_size):
	converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(
		model_path,
		output_arrays=['end_logits', 'start_logits'],
	)
	if reduce_model_size:
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	
	tflite_buffer = converter.convert()
	open(lite_model_path, 'wb').write(tflite_buffer)

	print('TFLite model created.')


if __name__ == '__main__':
	model_out = 'bert_qa_pt.tflite'
	model_path = input("Enter model output path: ")
	convert_model_to_tflite(model_path, model_out, True)
