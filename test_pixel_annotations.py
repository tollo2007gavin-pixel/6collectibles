import json
import re
import unittest
from pixel_annotations import add_annotations


class PixelAnnotationPageTests(unittest.TestCase):
    def test_portable_editor_escapes_data_and_does_not_duplicate_photos(self):
        result=dict(number=1,listing='id</script>',source_sha256='abc',human_review={'notes':'a</script>'})
        page='<h1>Review</h1><div id="pixel-review-notes-controls"></div><section><h2>Card 1</h2><img src="data:image/png;base64,original"></section>'
        batch={'cards':[{'number':1,'images':[{'label':'Corner close-ups','src':'data:image/png;base64,original'}]}]}
        output=add_annotations(page,[result],batch)
        self.assertEqual(output.count('data:image/png;base64,original'),1)
        self.assertIn('pixel-editor-1',output)
        data=json.loads(re.search(r'<script id="pixel-annotation-state" type="application/json">(.*?)</script>',output,re.S)[1])
        self.assertEqual(data['cards'][0]['listing'],'id</script>')
        self.assertEqual(data['cards'][0]['annotations'],[])
        self.assertTrue(data['cards'][0]['previously_reviewed'])
        self.assertEqual(add_annotations(output,[result],batch),output)

    def test_missing_sheet_is_not_an_annotation_training_record(self):
        output=add_annotations('<h1>Review</h1>',[{'number':1,'status':'no_saved_corner_sheet'}],{'cards':[{'number':1,'images':[]}]})
        data=json.loads(re.search(r'<script id="pixel-annotation-state" type="application/json">(.*?)</script>',output,re.S)[1])
        self.assertEqual(data['cards'],[])


if __name__=='__main__':unittest.main()
