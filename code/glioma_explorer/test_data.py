"""Geometry, provenance and API checks; local-data tests require the downloaded data."""
import io
import unittest

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from .app import app
from .data import UCSF_ROOT, MU_ROOT, align, catalog, image_info, slice_array


class GeometryTests(unittest.TestCase):
    def test_anatomical_landmark_appears_on_correct_side(self):
        # A known right/anterior/superior landmark in canonical RAS coordinates.
        a = np.zeros((4, 5, 6), dtype=np.uint8)
        a[3, 4, 5] = 1
        self.assertEqual(slice_array(a, "axial", 5)[0, -1], 1)  # top A, right R
        self.assertEqual(slice_array(a, "coronal", 4)[0, -1], 1)  # top S, right R
        self.assertEqual(slice_array(a, "sagittal", 3)[0, -1], 1)  # top S, right A

    def test_resampling_preserves_discrete_segmentation_labels(self):
        a = np.zeros((8, 8, 8), dtype=np.uint8)
        a[2:6, 2:6, 2:6] = 3
        affine = np.eye(4)
        affine[0, 3] = .4
        m = nib.Nifti1Image(a, affine)
        ref = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4))
        aligned = align(m, ref, mask=True)
        self.assertEqual(set(np.unique(aligned.dataobj)), {0, 3})
        np.testing.assert_allclose(aligned.affine, ref.affine)


@unittest.skipUnless(UCSF_ROOT.is_dir(), "Extracted UCSF directory is required")
class LocalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_cohort_counts_keep_different_record_types_separate(self):
        c = catalog()
        self.assertEqual((c["ucsf"]["patient_count"], c["ucsf"]["visits"]), (298, 596))
        self.assertTrue(all(p["image_available"] for p in c["ucsf"]["patients"]))
        self.assertEqual(c['ucsf']['storage'], 'extracted_directory')
        self.assertEqual((c["mu"]["patient_count"], c["mu"]["visits"], c["mu"]["scanner_rows"]), (203, 597, 654))
        self.assertTrue(c["mu"]["image_available"])
        self.assertEqual((c['mu']['image_visits'], c['mu']['mask_visits'], c['mu']['file_count']), (596, 594, 2978))
        self.assertEqual(c["mu"]["pairs"], 395)
        self.assertEqual(sum(x["count"] for x in c["mu"]["gap_histogram"]), 395)

    def test_mask_volumes_match_independently_inspected_original_table(self):
        info = image_info("100001")
        self.assertAlmostEqual(info["volumes_ml"][0]["2"], 21.178, places=3)
        self.assertAlmostEqual(info["volumes_ml"][0]["3"], .847, places=3)
        self.assertAlmostEqual(info["volumes_ml"][1]["2"], 18.069, places=3)

    def test_image_api_and_invalid_inputs(self):
        r = self.client.get('/api/slice/100004.png?index=94&overlay=false')
        self.assertEqual(r.status_code, 200)
        img = np.asarray(Image.open(io.BytesIO(r.content)))
        self.assertTrue(np.array_equal(img[..., 0], img[..., 1]))
        r2 = self.client.get('/api/slice/100004.png?index=94&overlay=true')
        self.assertNotEqual(r.content, r2.content)
        for url, code in [('/api/slice/100004.png?index=9999',422),
                          ('/api/slice/100004.png?timepoint=3',422),
                          ('/api/slice/100004.png?plane=unknown',422),
                          ('/api/slice/unknown.png',404),
                          ('/api/patient/mu/PatientID_0003',200)]:
            self.assertEqual(self.client.get(url).status_code, code, url)

    def test_mu_nonconsecutive_timepoints_and_missing_annotations(self):
        r = self.client.get('/api/patient/mu/PatientID_0003?first=2&second=5')
        self.assertEqual(r.status_code, 200)
        p = r.json()
        self.assertEqual([v['timepoint'] for v in p['image_visits']], [1, 2, 5])
        self.assertEqual(p['image']['timepoints'], [2, 5])
        self.assertEqual(p['image']['labels'][0]['name'], 'NETC')
        r = self.client.get('/api/slice/PatientID_0003.png?first=2&second=5&timepoint=5&index=94')
        self.assertEqual(r.status_code, 200)
        self.assertGreater(Image.open(io.BytesIO(r.content)).width, 100)
        self.assertEqual(self.client.get('/api/patient/mu/PatientID_0003?first=3').status_code, 422)
        self.assertEqual(self.client.get('/api/patient/mu/PatientID_0004').json()['image']['timepoints'], [1])
        # These MRI exist in both TCIA and the mirror, but their mask files do not.
        missing = self.client.get('/api/patient/mu/PatientID_0191?first=1').json()['image']
        self.assertEqual(missing['mask_available'], [False])
        self.assertEqual(missing['volumes_ml'], [None])
        self.assertEqual(missing['label_values'], [None])
        base = '/api/slice/PatientID_0191.png?first=1&timepoint=1&index=80'
        plain = self.client.get(base + '&overlay=false')
        self.assertEqual(plain.status_code, 200)
        self.assertEqual(plain.content, self.client.get(base + '&overlay=true').content)


if __name__ == '__main__':
    unittest.main()
