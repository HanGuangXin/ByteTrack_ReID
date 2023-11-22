import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):  # todo: ReID. add inputs of 'temp_feat', 'buffer_size'

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # TODO: add the following values and functions
        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    # TODO: ReID. for update embeddings during tracking
    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.update_features(new_track.curr_feat)       # TODO: added 20220322
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):     # TODO: 'update_feature' added 02003022
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:                  # TODO: added 20220322
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class FairMOTTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # [hgx 0119], to see contribution of each module in trakcing
        self.embedding_matched_percent = []
        self.IoU_matched_percent = []
        self.LowScore_matched_percent = []

    def update(self, output_results, img_info, img_size, id_feature):  # TODO: ReID. add 'id_feature'
        """
        update tracks, e.g. activated, refind, lost and removed tracks
        Args:
            output_results: tensor of shape [bbox_num, 7 + 128], 7 for bbox(4) + obj_conf + cls_conf + cls, 128 for embedding
            img_info:list, [origin_H, origin_W, 1, 1 img_path]
            img_size: tuple, (input_H, input_W)

        Returns:

        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:  # goes here
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]  # score = obj_conf * cls_conf
            bboxes = output_results[:, :4]  # x1y1x2y2, e.g. tlbr
        img_h, img_w = img_info[0], img_info[1]  # origin image size
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))  # input/origin, <1
        bboxes /= scale  # map bbox to origin image size

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)  # self.args.track_thresh > score > 0.1, for second matching
        dets_second = bboxes[inds_second]  # detections for second matching
        dets = bboxes[remain_inds]  # detections for first matching, high quality
        scores_keep = scores[remain_inds]
        id_feature_keep = id_feature[remain_inds]  # TODO: ReID. ID feature of 1st stage matching
        scores_second = scores[inds_second]
        id_feature_second = id_feature[inds_second]  # TODO: ReID. ID feature of 2nd stage matching

        if len(dets) > 0:
            '''Detections'''  # TODO: 'f, 30' for ReID
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 30) for
                          # detections to STracks, list [OT_0_(0-0), ...]
                          (tlbr, s, f) in
                          zip(dets, scores_keep, id_feature_keep)]  # class STrack in yolox/tracker/byte_tracker.py
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)  # tracked but bot confirmed, e.g. newly init tracks
            else:
                tracked_stracks.append(track)

        ''' Step 1: First association, with embedding'''  # TODO: borrowed from FairMOT
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool,
                                     detections)  # kalman filter with maha distance doing gating
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)  # self.args.match_thresh

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)  # IoU distance. 1 - _ious
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)  # refine 'dists' with detection scores
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)  # Hungarain

        for itracked, idet in matches:  # do update w.r.t matching results
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet],
                             self.frame_id)  # update of class STrack, update Kalman Filter, track state and other settings
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ''' Step 3: Second association, with low score detection boxes'''
        # # association the untrack to the low score detections
        # if len(dets_second) > 0:  # TODO: 'f, 30' for ReID
        #     '''Detections'''
        #     detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 30) for  # detections_second  --> STrack
        #                          (tlbr, s, f) in zip(dets_second, scores_second, id_feature_second)]
        # else:
        #     detections_second = []
        # second_tracked_stracks = [r_tracked_stracks[i] for i in u_track if
        #                           r_tracked_stracks[i].state == TrackState.Tracked]  # TODO: why Tracked?
        # dists = matching.iou_distance(second_tracked_stracks, detections_second)  # IoU distance. 1 - _ious
        # matches, u_track, u_detection_second = matching.linear_assignment(dists,
        #                                                                   thresh=0.5)  # Hungarain with lower thresh
        #
        # for itracked, idet in matches:  # do update w.r.t second matching results
        #     track = second_tracked_stracks[itracked]
        #     det = detections_second[idet]
        #     if track.state == TrackState.Tracked:
        #         track.update(det, self.frame_id)
        #         activated_starcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_stracks.append(track)

        for it in u_track:  # set unmatched tracks as 'lost'
            track = second_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]  # unmatched detections
        dists = matching.iou_distance(unconfirmed, detections)  # IoU distance. 1 - _ious
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)  # refine 'dists' with detection scores
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
