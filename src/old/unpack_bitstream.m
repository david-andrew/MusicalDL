function stream = unpack_bitstream(packed_stream)
%unpack_bitstream converts a packed integer stream to a logical array
%   packed stream is a uint array (valid widths are 8, 16, 32, and 64)
%   stream is the packed_stream converted to a logical array
%  
%   this function is the inverse of pack_bitstream() function
%   i.e. unpack_bitstream(pack_bitstream(stream)) == stream


    %convert all streams into a uint8 packed stream
    packed_stream = typecast(packed_stream, 'uint8');
    
    %create a container for the unpacked stream
    stream = zeros(length(packed_stream) * 8, 1, 'logical');
    
    %for every byte, sequentially place them into the stream
    for i = 1:length(packed_stream)
        byte = packed_stream(i);    %current byte to pack
        stream((i-1)*8+1:i*8) = bitget(byte, 1:8);
    end
end