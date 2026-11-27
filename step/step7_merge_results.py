def merge_segments(segments_per_chunk, gap_tolerance):
    """
    모든 조각의 결과를 시간순으로 모으고, 겹치거나 gap_tolerance 초 안으로 붙은 구간은 하나로 합친다.
    조각 경계에 걸친 장면은 양쪽 에이전트가 반씩 보고하기 때문에 이어 붙여야 한다.
    """
    all_segments = []
    for segments in segments_per_chunk:
        all_segments.extend(segments)
    all_segments.sort(key=lambda segment: segment.start)

    merged = []
    for segment in all_segments:
        if len(merged) == 0:
            merged.append(segment)
            continue

        previous = merged[-1]
        if segment.start > previous.end + gap_tolerance:
            merged.append(segment)
            continue

        previous.end = max(previous.end, segment.end)
        previous.reasons.extend(segment.reasons)
        for chunk_number in segment.chunk_numbers:
            if chunk_number not in previous.chunk_numbers:
                previous.chunk_numbers.append(chunk_number)

    return merged
