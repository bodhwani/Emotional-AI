import sys
import os
from models import prepare_database, extract_segment_level, extract_utterance_level


ROOT_PATH = os.getcwd() 
AUDIO_DIR = "datasets/IEMOCAP/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/Session"
MOD_AUDIO_DIR = "datasets/IEMOCAP/IEMOCAP_ahsn_leave-two-speaker-out"
MODEL_NAME='SER-DNN-ELM-Model'

ORIGINAL_DATASETS_PATH = os.path.join(ROOT_PATH, AUDIO_DIR)
MODIFIED_DATASETS_PATH = os.path.join(ROOT_PATH, MOD_AUDIO_DIR)

if __name__ == '__main__':
	nFolders=5
	segmentNum=25
	prepare_database.main05(ROOT_PATH, ORIGINAL_DATASETS_PATH, MODIFIED_DATASETS_PATH,MODEL_NAME)
	extract_segment_level.main(ROOT_PATH,MODIFIED_DATASETS_PATH,MODEL_NAME,nFolders,segmentNum)
	extract_utterance_level.main(ROOT_PATH,MODEL_NAME,nFolders)
