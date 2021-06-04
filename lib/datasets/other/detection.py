class Detection(object):

    def __init__(self, bbox, bbox_normalized=None, confidence=0, crop=None, track_id=-1, format='tlbr'):
        self.bbox = bbox
        self.box_format = format
        self.bbox_normalized = bbox_normalized
        self.confidence = float(confidence)
        self.crop = crop
        self.track_id = track_id

    def to_tlwh(self):
        ret = self.bbox.copy()
        if self.box_format == 'tlwh':  # górny lewy róg, szerokość i wysokość
            pass
        elif self.box_format == 'tlbr':  # górny lewy róg i dolny prawy
            # ret[0] = self.box[0]
            # ret[1] = self.box[1]
            ret[2] = self.bbox[2] - self.bbox[0]
            ret[3] = self.bbox[3] - self.bbox[1]
        elif self.box_format == 'xyah':  # środek ramki, stosunek szerokość / wysokość i wysokość
            # ret[3] = self.box[3]
            ret[2] = self.bbox[2] * self.bbox[3]
            ret[0] = self.bbox[0] - ret[2] / 2
            ret[1] = self.bbox[1] - self.bbox[3] / 2
        elif self.box_format == 'xywh':  # środek ramki, szerokość i wysokość
            # ret[2] = self.box[2]
            # ret[3] = self.box[3]
            ret[0] = self.bbox[0] - self.bbox[2] / 2
            ret[1] = self.bbox[1] - self.bbox[3] / 2
        return ret

    def to_tlbr(self):
        ret = self.bbox.copy()
        if self.box_format == 'tlwh':  # górny lewy róg, szerokość i wysokość
            ret[2] = self.bbox[2] + self.bbox[0]
            ret[3] = self.bbox[3] + self.bbox[1]
        elif self.box_format == 'tlbr':  # górny lewy róg i dolny prawy
            pass
        elif self.box_format == 'xyah':  # środek ramki, stosunek szerokość / wysokość i wysokość
            ret[1] = self.bbox[1] - self.bbox[3] / 2
            ret[3] = self.bbox[1] + self.bbox[3] / 2
            ret[0] = self.bbox[0] - (self.bbox[2] * self.bbox[3]) / 2
            ret[2] = self.bbox[0] + (self.bbox[2] * self.bbox[3]) / 2
        elif self.box_format == 'xywh':  # środek ramki, szerokość i wysokość
            ret[1] = self.bbox[1] - self.bbox[3] / 2
            ret[3] = self.bbox[1] + self.bbox[3] / 2
            ret[0] = self.bbox[0] - self.bbox[2] / 2
            ret[2] = self.bbox[0] + self.bbox[2] / 2
        return ret

    def to_xyah(self):
        ret = self.bbox.copy()
        if self.box_format == 'tlwh':  # górny lewy róg, szerokość i wysokość
            ret[2] = self.bbox[2] / self.bbox[3]
            ret[0] = self.bbox[0] + self.bbox[2] / 2
            ret[1] = self.bbox[1] + self.bbox[3] / 2
        elif self.box_format == 'tlbr':  # górny lewy róg i dolny prawy
            ret[0] = (self.bbox[0] + self.bbox[2]) / 2
            ret[1] = (self.bbox[1] + self.bbox[3]) / 2
            ret[3] = self.bbox[3] - self.bbox[1]
            ret[2] = (self.bbox[2] - self.bbox[0]) / ret[3]
        elif self.box_format == 'xyah':  # środek ramki, stosunek szerokość / wysokość i wysokość
            pass
        elif self.box_format == 'xywh':  # środek ramki, szerokość i wysokość
            ret[2] = self.bbox[2] / self.bbox[3]
        return ret

    def to_xywh(self):
        ret = self.bbox.copy()
        if self.box_format == 'tlwh':  # górny lewy róg, szerokość i wysokość
            ret[0] = self.bbox[0] + self.bbox[2] / 2
            ret[1] = self.bbox[1] + self.bbox[3] / 2
        elif self.box_format == 'tlbr':  # górny lewy róg i dolny prawy
            ret[0] = (self.bbox[0] + self.bbox[2]) / 2
            ret[1] = (self.bbox[1] + self.bbox[3]) / 2
            ret[3] = self.bbox[3] - self.bbox[1]
            ret[2] = self.bbox[2] - self.bbox[0]
        elif self.box_format == 'xyah':  # środek ramki, stosunek szerokość / wysokość i wysokość
            ret[2] = self.bbox[2] * self.bbox[3]
        elif self.box_format == 'xywh':  # środek ramki, szerokość i wysokość
            pass
        return ret

    def change_format(self, to_format):
        if to_format == 'tlwh':
            self.bbox = self.to_tlwh()
            self.box_format = 'tlwh'
        elif to_format == 'tlbr':
            self.bbox = self.to_tlbr()
            self.box_format = 'tlbr'
        elif to_format == 'xyah':
            self.bbox = self.to_xyah()
            self.box_format = 'xyah'
        elif to_format == 'xywh':
            self.bbox = self.to_xywh()
            self.box_format = 'xywh'
        else:
            print(f"[!] Nie ma takiego formatu dla detekcji: {to_format}")
