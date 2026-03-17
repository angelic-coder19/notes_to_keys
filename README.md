# RESULTS OF PREPROCESSING:
Processing files: 100%|█████████████████████████████████████████████████████████| 132/132 [10:28<00:00,  4.76s/it] 
<br>INFO - Dataset loaded: 71741 samples
<br>INFO -   CQT shape: (71741, 1, 176, 100)
<br>INFO -   Onset shape: (71741, 100, 88)
<br>INFO -   Frame shape: (71741, 100, 88)
<br>INFO - Saved test data to processed_chunks\test_data.npz
<br>INFO - ✅ Processing complete!

### The above results are obtained from only the 2004 folder recordings containg 71741 samples

## DATA VISUALIZATION:
![Sample cqt, onset, and frame data](validation_plots\sample_33214_visualization.png)

![Distribution of note onsets across the dataset](validation_plots\samples_overview.png)

The above visualizations show dislay:
- Normalized inputs (colors -2 to 1.5, centered around 0)
- Binary targets (onset and frames are 0 or 1)
- Musical structer visible (horizontal bands=notes)
- Variety accross smaples (not all are identical)
- Reasonable sparsity (not too many/few notes)
- Apparent alignemnt (onsets match CQT stars)

The conclustion from the above is that the preprocessing pipeline is working correctly, producing well-structured data that captures the musical information needed for training a model. The visualizations confirm that the CQT inputs and corresponding targets are properly aligned and exhibit expected patterns, indicating that the dataset is ready for use in training a music transcription model.