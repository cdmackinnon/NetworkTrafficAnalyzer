import pandas as pd

def read_and_filter_data(file_path):
    data = pd.read_csv(file_path)
    # Drop unused columns
    filtered = data.drop(columns=['Source', 'Destination', 'Info', 'No.'])
    return filtered

def calculate_protocol_percentages(filtered):
    protocol_counts = filtered['Protocol'].value_counts()
    protocol_percentages = (protocol_counts / len(filtered))

    # Initialize variables with default values
    TCP_percentage = 0
    TLS_percentage = 0
    UDP_percentage = 0
    QUIC_percentage = 0

    # Extract percentages for TCP, TLS, and UDP protocols if available
    try:
        TCP_percentage = protocol_percentages['TCP']
    except KeyError:
        pass
    try:
        TLS_percentage = protocol_percentages['TLSv1.3']
    except KeyError:
        pass
    try:
        UDP_percentage = protocol_percentages['UDP']
    except KeyError:
        pass
    try:
        QUIC_percentage = protocol_percentages['QUIC']
    except KeyError:
        pass
    return TCP_percentage, TLS_percentage, UDP_percentage, QUIC_percentage

def calculate_average_bytes_per_second(filtered):
    average_bytes_per_second = filtered['Length'].sum() / filtered['Time'].max()
    return average_bytes_per_second

def calculate_average_packets_per_second(filtered):
    average_packets_per_second = len(filtered) / filtered['Time'].max()
    return average_packets_per_second

def calculate_packet_length_statistics(filtered):
    # Packet Length Stats
    packet_mode_length = filtered['Length'].mode()[0]
    packet_mean_length = filtered['Length'].mean()
    packet_median_length = filtered['Length'].median()
    packet_std_length = filtered['Length'].std()
    return packet_mode_length, packet_mean_length, packet_median_length, packet_std_length

def calculate_packet_timing_statistics(filtered):
    # Compute time differences between consecutive packets
    packet_timings = filtered['Time'].diff()
    # Calculate statistics
    packet_mean_timing = packet_timings.mean()
    packet_median_timing = packet_timings.median()
    packet_std_timing = packet_timings.std()
    return packet_mean_timing, packet_median_timing, packet_std_timing

def generate_statistics_dictionary(filtered, website):
    # Calculate protocol percentages
    TCP_percentage, TLS_percentage, UDP_percentage, QUIC_percentage = calculate_protocol_percentages(filtered)
    # Calculate average bytes per second
    average_bytes_per_second = calculate_average_bytes_per_second(filtered)
    # Calculate average packets per second
    average_packets_per_second = calculate_average_packets_per_second(filtered)
    # Calculate packet length statistics
    packet_mode_length, packet_mean_length, packet_median_length, packet_std_length = calculate_packet_length_statistics(filtered)
    # Calculate packet timing statistics
    packet_mean_timing, packet_median_timing, packet_std_timing = calculate_packet_timing_statistics(filtered)
    
    statistics_dict = {
        'packet_mean_timing': packet_mean_timing,
        'packet_median_timing': packet_median_timing,
        'packet_std_timing': packet_std_timing,
        'packet_mode_length': packet_mode_length,
        'packet_mean_length': packet_mean_length,
        'packet_median_length': packet_median_length,
        'packet_std_length': packet_std_length,
        'average_packets_per_second': average_packets_per_second,
        'average_bytes_per_second': average_bytes_per_second,
        'TCP_percentage': TCP_percentage,
        'TLS_percentage': TLS_percentage,
        'UDP_percentage': UDP_percentage,
        'QUIC_percentage':QUIC_percentage,
        'Website': website
    }
    
    return statistics_dict

def main(file_path, website):
    filtered = read_and_filter_data(file_path)
    statistics_dict = generate_statistics_dictionary(filtered, website)
    statistics_df = pd.DataFrame(statistics_dict, index=[0])
    return statistics_df
