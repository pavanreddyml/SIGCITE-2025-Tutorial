import os
import json

import numpy as np
from PIL import Image


class AdversarialSelector:
    def __init__(self, root_path):
        self.root_path = root_path

    def attacker_selector(self, config):
        import tensorflow as tf
        from adversarial_lab.arsenal.adversarial.whitebox import FastSignGradientMethodAttack, ProjectedGradientDescentAttack, CarliniWagnerAttack, DeepFoolAttack, SmoothFoolAttack
        from adversarial_lab.arsenal.adversarial.blackbox import FiniteDifferenceAttack, NESAttack, RGFAttack, SPSAAttack
        category = config["category"]
        model_name = config["model"]

        attack = config["attack"]
        attack_params = config["attack_params"]

        image_array = config.get("image_array", None)
        image_class = config["image_class"]
        image_name = config["image_name"]
        target_class_idx = config["target_class_idx"]



        # Get all model related data
        model_data = self.get_model(model_name)
        model = model_data["model"]
        preprocess_fn = model_data["preprocess_input"]
        decode_preds = model_data["decode_predictions"]
        input_shape = model_data["input_shape"]



        # Get image Array
        dataset_dir_map = {
            "inception": "imagenet",
            "resnet": "imagenet",
            "mobilenet": "imagenet",
            "mnist_digits": "digits"
        }
        if image_array is not None:
            img_arr = image_array
            img_arr = tf.image.resize(img_arr, input_shape[:2]).numpy()
        else:
            img_arr = self.get_image(directory=dataset_dir_map[model_name],
                                     image_class=image_class,
                                     image_name=image_name,
                                     input_shape=input_shape)


        # Get Attacker
        if category == "whitebox":
            def preprocess(sample, *args, **kwargs):
                input_sample = tf.cast(sample, dtype=tf.float32)
                if len(input_sample.shape) == 2:
                    input_sample = tf.expand_dims(input_sample, axis=-1)
                    input_sample = tf.image.grayscale_to_rgb(input_sample)

                elif len(input_sample.shape) == 3 and input_sample.shape[-1] == 1:
                    input_sample = tf.image.grayscale_to_rgb(input_sample)

                input_tensor = tf.convert_to_tensor(input_sample, dtype=tf.float32)
                resized_image = tf.image.resize(input_tensor, input_shape[:2])
                batch_image = tf.expand_dims(resized_image, axis=0)
                return preprocess_fn(batch_image)

            if attack == "Fast Sign Gradient Method":
                attacker = FastSignGradientMethodAttack
            elif attack == "Projected Gradient Descent":
                attacker = ProjectedGradientDescentAttack
            elif attack == "Carlini Wagner":
                attacker = CarliniWagnerAttack
            elif attack == "Deep Fool":
                attacker = DeepFoolAttack
            elif attack == "Smooth Fool":
                attacker = SmoothFoolAttack

            from tensorflow.keras.utils import get_file
            path = get_file(
                "imagenet_class_index.json",
                origin="https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
            )
            with open(path, "r") as f:
                mapping = json.load(f)

            def decode_top(preds, top=5):
                mapping = globals().get("IMAGENET_CLASS_INDEX")
                if mapping is None:
                    # Fallback: load keras mapping
                    from tensorflow.keras.utils import get_file
                    path = get_file(
                        "imagenet_class_index.json",
                        origin="https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
                    )
                    with open(path, "r") as f:
                        mapping = json.load(f)
                wnid_to_idx = {v[0]: int(k) for k, v in mapping.items()}

                out = []
                for sample in decode_preds(preds, top=top):
                    rows = []
                    for wnid, name, conf in sample:
                        arr_idx = wnid_to_idx.get(str(wnid), -1)
                        rows.append((arr_idx, str(name), float(conf)))
                    out.append(rows)
                return out


            return {
                "attacker": attacker,
                "model": model,
                "preprocess": preprocess,
                "attack_params": attack_params,
                "input_shape": input_shape,
                "sample": img_arr,
                "decode_preds": decode_top,
                "target_class": target_class_idx,
                "class_dict": mapping
            }

        if category == "blackbox":
            if attack == "Finite Difference":
                attacker = FiniteDifferenceAttack
            elif attack == "NES":
                attacker = NESAttack
            elif attack == "RGF":
                attacker = RGFAttack
            elif attack == "SPSA":
                attacker = SPSAAttack

            return {
                "attacker": attacker,
                "model": model,
                "preprocess": None,
                "attack_params": attack_params,
                "input_shape": input_shape,
                "sample": img_arr.astype(int),
                "decode_preds": decode_preds,
                "target_class": target_class_idx,
                "class_dict": {str(i): str(i) for i in range(10)}
            }

    def get_model(self, dataset):

        if dataset == "inception":
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
            model = InceptionV3(weights='imagenet')
            input_shape = (299, 299, 3)
            return {
                "model": model,
                "preprocess_input": preprocess_input,
                "decode_predictions": decode_predictions,
                "input_shape": input_shape
            }
        elif dataset == "resnet":
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
            model = ResNet50(weights='imagenet')
            input_shape = (224, 224, 3)
            return {
                "model": model,
                "preprocess_input": preprocess_input,
                "decode_predictions": decode_predictions,
                "input_shape": input_shape
            }
        elif dataset == "mobilenet":
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
            model = MobileNetV2(weights='imagenet')
            input_shape = (224, 224, 3)
            return {
                "model": model,
                "preprocess_input": preprocess_input,
                "decode_predictions": decode_predictions,
                "input_shape": input_shape
            }
        elif dataset == "mnist_digits":
            import tensorflow as tf
            model = tf.keras.models.load_model(os.path.join(self.root_path, "assets/models/mnist_digits.h5"))

            def pred_fn(samples, *args, **kwargs):
                x = np.asarray(samples, dtype=np.float32)

                # Convert RGB -> grayscale using NumPy only
                if x.ndim == 4 and x.shape[-1] == 3:              # (N, 28, 28, 3) or (1, 28, 28, 3)
                    x = x.mean(axis=-1)
                elif x.ndim == 3 and x.shape[0] == 3 and x.shape[1:] == (28, 28):  # (3, 28, 28)
                    x = x.mean(axis=0, keepdims=True)

                # Ensure shape (N, 28, 28)
                if x.ndim == 2 and x.shape == (28, 28):           # single image
                    x = x.reshape(1, 28, 28)
                elif x.ndim == 3 and x.shape[-2:] == (28, 28):    # already batched 2D
                    x = x.reshape(-1, 28, 28)
                else:
                    # Last-resort reshape if dimensions are compatible
                    x = x.reshape(-1, 28, 28)

                x = x.astype(np.float32)

                preds = model.predict(x, verbose=0)
                return [pred for pred in preds]

            def decode_predictions(preds, top=5):
                results = []
                for pred in preds:
                    top_indices = np.argsort(pred)[-top:][::-1]
                    top_scores = [(str(i), str(i), float(pred[i])) for i in top_indices]
                    results.append(top_scores)
                return results

            input_shape = (28, 28, 1)
            return {
                "model": pred_fn,
                "preprocess_input": None,
                "decode_predictions": decode_predictions,
                "input_shape": input_shape
            }

    def get_image(self, directory, image_class, image_name, input_shape):
        path = os.path.join(self.root_path, "assets/images", directory, image_class, image_name)
        img = Image.open(path).convert("RGB")
        img = img.resize((input_shape[1], input_shape[0]))
        img_arr = np.array(img)
        return img_arr
