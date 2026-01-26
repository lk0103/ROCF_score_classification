import json
import os
import re

from GeneralResNetTraining import GeneralResNetTraining

from ROCFDataset_for_CNN import LoadROCFDataset


class LlmEvaluation():
    def __init__(self, llm_results_file_path=''):
        self.llm_results_file_path = llm_results_file_path

    def llm_testing(self):
        test_name = 'LLM_few_shot_class'
        f = f'{test_name}.txt'
        # Load your dataset
        ROCF_dataset = LoadROCFDataset()

        general_resnet_training = GeneralResNetTraining(f=f)
        transform = general_resnet_training.get_resnet_transforms()
        val_test_transform = general_resnet_training.get_resnet_transforms(default=True)

        _, _, test_loader = general_resnet_training.initialize_datasets(
            rocf_dataset=ROCF_dataset, transform=transform, val_test_transform=val_test_transform
        )

        test_results = self.parse_llm_test_results()

        # Evaluation
        print("GPT 5.2 thinking MODEL Accuracy on testing set: ")
        general_resnet_training.test(model=None, dataloader=test_loader, loss_fn=None,
                                     prefix='GPT 5.2 thinking MODEL Test', test_results=test_results)

    def parse_llm_test_results(self):
        test_file_paths_def = (
            'C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/'
            'orezane_1500x1500px/Train_Val_Test_split/test_len_1041_4_classes.txt'
        )

        # Load full test image paths
        with open(test_file_paths_def, 'r', encoding='utf-8') as f:
            test_image_paths = [line.strip() for line in f if line.strip()]

        # Build mapping:
        #   image_name_without_score -> full_image_path
        image_name_to_full_path = {}
        for path in test_image_paths:
            base = os.path.basename(path)
            name_no_ext = os.path.splitext(base)[0]

            # Remove score suffix (_12,5, _22, etc.)
            name_core = re.sub(r'_[0-9]+([.,][0-9]+)?$', '', name_no_ext)

            image_name_to_full_path[name_core] = path

        # Parse LLM results file (multiple JSON objects)
        results = {}

        with open(self.llm_results_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Each result is a standalone JSON object
        json_objects = re.findall(r'\{.*?\}', content, flags=re.DOTALL)

        for obj_str in json_objects:
            data = json.loads(obj_str)

            image_name = data["image_name"]  # e.g. CM17SG02_3.jpg
            class_int = data["class_int"]

            image_core = os.path.splitext(image_name)[0]

            if image_core not in image_name_to_full_path:
                raise KeyError(
                    f"Image '{image_core}' not found in test file paths definition."
                )

            full_image_path = image_name_to_full_path[image_core]
            results[full_image_path] = class_int

        return results


# Example usage:
trainer = LlmEvaluation(llm_results_file_path='C:/Users/lucin/OneDrive/Desktop/diplomovka/thesis_code/prompts/LLM_few_shot_class/few_shot_results.txt')
trainer.llm_testing()