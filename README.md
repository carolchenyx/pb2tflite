# pb2tflite
tensorflow model to tflite model

## example code

    import tensorflow as tf

    in_path = "resnet18False2021-02-25-10-20-39.pb"
    out_path = "resnet18False.tflite"


    # input node name
    input_tensor_name = ["Image"]
    input_tensor_shape = {"Image":[1,224,224,3]}
    # output node name
    classes_tensor_name = ["network/Output"]

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name, classes_tensor_name, input_shapes=input_tensor_shape)

    # the following is the option u can choose while u can not convert the pb model
    # converter.allow_custom_ops=True
    # converter.experimental_new_converter =True
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    # converter.post_training_quantize = True

    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()

    with open(out_path, "wb") as f:
        f.write(tflite_model)
