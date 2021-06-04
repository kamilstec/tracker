class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(self, mean, cov, bbox, bbox_normalized, track_id, n_hits_to_init=3, max_age=30, crop=None, cnn_embedding=None):
        self.mean = mean
        self.covariance = cov
        self.bbox = bbox  # położenie ramki / położenie ścieżki na ostatniej klatce (oryginalne koordynaty)
        self.bbox_normalized = bbox_normalized  # znormalizowane położenie ramki (0-1, gdzie 0 to jeden róg,
        # a 1 to przeciwległy róg)
        self.crop = crop  # wycinek przedstawiający daną osobę z ostatniej klatki, na której przypisano do tej ścieżki
        # nową detekcję
        #self.last_frame_id = frame_id  # klatka, na której ostatnio przypisano detekcję do tej ścieżki
        self.cnn_embedding = cnn_embedding
        self.track_id = track_id  # niepowtarzalne id ścieżki
        self.hits = 1
        self.age = 1
        self.time_since_update = 0  # ilość klatek, które minęły od ostatniego przypisania detekcji do tej ścieżki
        self.state = TrackState.Tentative

        self.max_age = max_age

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())

        self.time_since_update = 0
        self.hits += 1
        if self.state == TrackState.Tentative and self.hits >= 3:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update >= self.max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_lost(self):
        if self.time_since_update > 1:
            return True
        return False
