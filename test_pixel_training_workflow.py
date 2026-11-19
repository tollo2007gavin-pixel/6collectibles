import tempfile
import unittest
from pathlib import Path
from pixel_training_workflow import settings


class TrainingWorkflowTests(unittest.TestCase):
    def test_development_default_and_frozen_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'review_labels').mkdir()
            self.assertEqual(settings({},root)[0],'baseline')
            with self.assertRaises(ValueError):settings({'mode':'validate'},root)
            (root/'frozen.json').write_text('{}')
            artifact=root/'tune_config'/'outputs'/'pixel_training'/'one';artifact.mkdir(parents=True)
            (artifact/'frozen-config.json').write_text('{}')
            mode,_,config=settings({'mode':'validate','frozen_config':'frozen.json'},root)
            self.assertEqual(mode,'validate');self.assertEqual(config,root/'frozen.json')
            mode,_,downloaded=settings({'mode':'validate','tune_run_id':'12345'},root)
            self.assertEqual(downloaded,root/'tune_config')
            with self.assertRaises(ValueError):settings({'mode':'validate','tune_run_id':'abc'},root)
            with self.assertRaises(ValueError):settings({'mode':'validate','tune_run_id':'123','frozen_config':'frozen.json'},root)

    def test_no_arbitrary_modes_or_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            for values in ({'mode':'run-api'},{'dataset_path':'../'}, {'dataset_path':'missing'}):
                with self.subTest(values=values),self.assertRaises(ValueError):settings(values,Path(tmp))


if __name__=='__main__':unittest.main()
