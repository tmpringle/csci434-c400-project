# this preprocessor works on one folder per run, so you need to rename the folder
# you're working in and the label each stream uses
#
# once each website's streams are summarized, you still need to manually compile
# them all into a single CSV

import pandas as pd
import sys

features = ["Label", "Packet Count", "Total Length", "Average Packet Interval",
            "Maximum Packet Interval", "Minimum Packet Interval", "Average Packet Length",
            "Maximum Packet Length", "Minimum Packet Length", "Most Common Packet Length"]

# dataframe for summaries of all streams analyzed
summary_of_all_streams = pd.DataFrame(columns=features)

for i in range(1, 16):
    cur_data_stream = pd.read_csv(f'./Stack Overflow/stackoverflow_stream{i}.csv')

    # values summarizing current packet stream
    summary_of_cur_stream = ["Stack Overflow", None, None, None, None, None, None, None, None, None]

    # packet count
    summary_of_cur_stream[1] = cur_data_stream.count()["No."]

    # total length
    summary_of_cur_stream[2] = cur_data_stream["Length"].sum()

    # packet info
    # avg packet interval and length
    total_time = cur_data_stream["Time"][summary_of_cur_stream[1] - 1]
    avg_pkt_interval = total_time / summary_of_cur_stream[1]

    # avg pkt interval
    summary_of_cur_stream[3] = avg_pkt_interval

    # avg pkt length
    summary_of_cur_stream[6] = summary_of_cur_stream[2] / summary_of_cur_stream[1]

    # min and max packet intervals and lengths
    max_pkt_interval = 0
    min_pkt_interval = float("inf")

    max_pkt_len = 0
    min_pkt_len = sys.maxsize

    for i in range(1, summary_of_cur_stream[1]):
        cur_pkt_interval = cur_data_stream["Time"][i] - cur_data_stream["Time"][i - 1]

        if cur_pkt_interval > max_pkt_interval:
            max_pkt_interval = cur_pkt_interval
        if cur_pkt_interval < min_pkt_interval and cur_pkt_interval > 0:
            min_pkt_interval = cur_pkt_interval

        cur_pkt_len = cur_data_stream["Length"][i]

        if cur_pkt_len > max_pkt_len:
            max_pkt_len = cur_pkt_len
        if cur_pkt_len < min_pkt_len:
            min_pkt_len = cur_pkt_len

    summary_of_cur_stream[4] = max_pkt_interval
    summary_of_cur_stream[5] = min_pkt_interval
    summary_of_cur_stream[7] = max_pkt_len
    summary_of_cur_stream[8] = min_pkt_len

    # most common packet length
    summary_of_cur_stream[9] = cur_data_stream["Length"].mode()[0]

    # add row to dataframe for all streams
    summary_of_all_streams = pd.concat([summary_of_all_streams, pd.DataFrame([summary_of_cur_stream], columns=features)])

summary_of_all_streams.to_csv("./Stack Overflow/Summary/all_stackoverflow_streams.csv", index=False)