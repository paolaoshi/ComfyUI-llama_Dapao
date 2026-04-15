import struct


def read_u32(file_obj):
    return struct.unpack("<I", file_obj.read(4))[0]


def read_u64(file_obj):
    return struct.unpack("<Q", file_obj.read(8))[0]


def read_string(file_obj):
    length = read_u64(file_obj)
    return file_obj.read(length).decode("utf-8")


def read_value(file_obj):
    value_type = read_u32(file_obj)

    if value_type == 0:
        return struct.unpack("<B", file_obj.read(1))[0]
    if value_type == 1:
        return struct.unpack("<b", file_obj.read(1))[0]
    if value_type == 2:
        return struct.unpack("<H", file_obj.read(2))[0]
    if value_type == 3:
        return struct.unpack("<h", file_obj.read(2))[0]
    if value_type == 4:
        return struct.unpack("<I", file_obj.read(4))[0]
    if value_type == 5:
        return struct.unpack("<i", file_obj.read(4))[0]
    if value_type == 6:
        return struct.unpack("<f", file_obj.read(4))[0]
    if value_type == 7:
        return struct.unpack("<?", file_obj.read(1))[0]
    if value_type == 8:
        return read_string(file_obj)
    if value_type == 9:
        array_type = read_u32(file_obj)
        count = read_u64(file_obj)
        return [read_value_of_type(file_obj, array_type) for _ in range(count)]
    if value_type == 10:
        return struct.unpack("<Q", file_obj.read(8))[0]
    if value_type == 11:
        return struct.unpack("<q", file_obj.read(8))[0]
    if value_type == 12:
        return struct.unpack("<d", file_obj.read(8))[0]

    raise ValueError(f"Unknown GGUF value type: {value_type}")


def read_value_of_type(file_obj, array_type):
    if array_type == 0:
        return struct.unpack("<B", file_obj.read(1))[0]
    if array_type == 1:
        return struct.unpack("<b", file_obj.read(1))[0]
    if array_type == 2:
        return struct.unpack("<H", file_obj.read(2))[0]
    if array_type == 3:
        return struct.unpack("<h", file_obj.read(2))[0]
    if array_type == 4:
        return struct.unpack("<I", file_obj.read(4))[0]
    if array_type == 5:
        return struct.unpack("<i", file_obj.read(4))[0]
    if array_type == 6:
        return struct.unpack("<f", file_obj.read(4))[0]
    if array_type == 7:
        return struct.unpack("<?", file_obj.read(1))[0]
    if array_type == 8:
        return read_string(file_obj)
    if array_type == 10:
        return struct.unpack("<Q", file_obj.read(8))[0]
    if array_type == 11:
        return struct.unpack("<q", file_obj.read(8))[0]
    if array_type == 12:
        return struct.unpack("<d", file_obj.read(8))[0]

    raise ValueError(f"Unknown GGUF array item type: {array_type}")


def get_layer_count(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            if file_obj.read(4) != b"GGUF":
                raise ValueError("This is not a GGUF file")

            _version = read_u32(file_obj)
            _tensor_count = read_u64(file_obj)
            kv_count = read_u64(file_obj)
            metadata = {}

            for _ in range(kv_count):
                key = read_string(file_obj)
                value = read_value(file_obj)
                metadata[key] = value

        for key, value in metadata.items():
            if key.lower().endswith(".block_count"):
                return int(value)
    except Exception:
        return None

    return None
