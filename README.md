# FOMC-communication
Replication files for 'How can a Fed Chair not be an actor'? - Effects of verbal and non-verbal FOMC tone on asset prices


# Text sentiment analysis and main analysis

1. Data/Fed_text.xlsx file contain FOMC statements and press conference transcripts between 2011 and 2025
2. Run text_topic_class.py to assign topic, sentiment and uncertainty score to each textual record (baseline model is where joined=1, meaning that consecutive sentences belonging to the same topics are merged)
  - Output: text_classified_joined.xlsx

3. Run main_analysis.py (until row 74) to calculate sentiment indicators for each statement and press conference
4. The remaining content of main_analysis.py contains the sentiment shocks construction method and final regressions (still incomplete)
   - Inputs:
       - Data/conf_changes.xlsx and Data/ann_changes.xlsx contains asset prices changes in the statement and PC windows
       - Data/cieslak_ann.xlsx and Data/cieslak_conf.xlsx contains structural decompositions of asset prices changes in the statement and PC windows based on Cieslak and Pang (2021)
       - Data/voicetonescores.xlsx contain the voice tone score calculated (see method in the next section)
       - Data/controls_fomc.xlsx contains various control variables (read the paper contact me for more details)
  
# Speech emotion recognition

1. Run audio_manipulation.py to download FOMC press conference audio files from YouTube and cut them into segments based on Data/timestamps_raw.xlsx
2. Audio_emotion_recogntition.ipynb runs SER models on the segments generated above (recommende to run on a high-capacity server or Google Colab) 
  - Output: Emotion of each audio segment according to the chosen SER model. Data/voicetonescores.xlsx was created based on these outputs.



   
