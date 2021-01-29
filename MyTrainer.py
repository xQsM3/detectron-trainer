from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from LossEvalHook import LossEvalHook
import os
import logging
import json
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)                 
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))

        return hooks



'''
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            
        if not os.path.isfile(os.path.join(output_folder,dataset_name+'_coco_format.json')):
            with open(os.path.join(output_folder,dataset_name+'_coco_format.json'), 'w') as db_file:
                db_file.write(json.dumps({}))
           
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
'''
