import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import prepare_detector_data as data


class DetectorDataTests(unittest.TestCase):
    def fixture(self, root):
        sheet = Image.new('RGB',(74,114))
        for name,xy,color in [('TL',(0,20),'red'),('TR',(42,20),'blue'),
                              ('BL',(0,82),'green'),('BR',(42,82),'yellow')]:
            sheet.paste(color, (*xy,xy[0]+32,xy[1]+32))
        buf = io.BytesIO()
        sheet.save(buf,'PNG')
        raw = buf.getvalue()
        c = dict(number=4,listing='v1|123|0',group_id='physical',source_sha256=data.sha(raw),
                 split='validation',annotations=[dict(corner='TR',label='whitening',box=[2,3,4,5]),
                 dict(corner='TL',label='uncertain',box=[1,1,3,3])],notes='',complete_corners=[],
                 result=dict(measurement_image_size=[74,114]))
        path=root/'saved.html'
        doc=(f'<img src="data:image/png;base64,bm90IHBpeGVscw==" alt="Candidate light spots">'
             f'<img src="data:image/png;base64,{base64.b64encode(raw).decode()}" alt="Original saved corner sheet">'
             '<script id="pixel-annotation-state" type="application/json">'+json.dumps(dict(schema=data.SCHEMA,cards=[c]))+
             '</script><script>c.positive_only_review=Boolean(c.annotations.length);</script>')
        path.write_text(doc)
        return c,path

    def test_crop_exact_unmarked_tile_and_keep_related_exposures_out_of_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,review=self.fixture(root)
            labels=root/'labels';labels.mkdir()
            first=data.build([review],labels,root/'d1')
            self.assertEqual(first['counts']['test'],dict(corners=3,groups=1,chips=1))
            tr=next(r for r in first['images'] if r['corner']=='TR')
            self.assertEqual(tr['boxes'],[[2,3,6,8]])
            with Image.open(root/'d1'/tr['file']) as corner:
                self.assertEqual(corner.getpixel((3,7)),(0,0,255))
            self.assertEqual(tr['image_sha256'],data.sha((root/'d1'/tr['file']).read_bytes()))
            ledger=root/'exposure';ledger.mkdir()
            (ledger/'detector.json').write_text(json.dumps(dict(schema=data.EXPOSURE_SCHEMA,exposed_aliases=['listing:123'])))
            second=data.build([review],labels,root/'d2',ledger)
            self.assertEqual(second['counts']['test']['corners'],0)
            c.update(source_sha256='f'*64,split='development',group_id='older-group')
            (labels/'old-batch.json').write_text(json.dumps(dict(schema=data.SCHEMA,cards=[c])))
            third=data.build([review],labels,root/'d3')
            self.assertEqual(third['counts']['test']['groups'],0)
            self.assertIn('group:older-group',third['images'][0]['aliases'])

    def test_exclude_unreviewed_and_ambiguous_corners(self):
        c=dict(number=7,notes='not a card',annotations=[],complete_corners=[],positive_only_review=True)
        self.assertFalse(data.corner_labels(c,32))
        c['notes']=' No whitening. '
        self.assertEqual(set(data.corner_labels(c,32)),set(data.CORNERS))
        c.update(notes='',annotations=[dict(label='whitening',corner='BR',box=[1,1,2,3])],
                 positive_only_review=False)
        self.assertFalse(data.corner_labels(c,32))
        c['complete_corners']=['BR']
        self.assertEqual(data.corner_labels(c,32),{'BR':[[1,1,3,4]]})


if __name__=='__main__':
    unittest.main()
