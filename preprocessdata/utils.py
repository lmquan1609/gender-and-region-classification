def chunk(arr, chunk_len):
    for i in range(0, len(arr), chunk_len):
        yield arr[i: i + chunk_len]