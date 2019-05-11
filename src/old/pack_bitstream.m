function packed_stream = pack_bitstream(stream, width)
%pack_bitstream converts an array of booleans to a packed array of uint64s
%   stream is the array of booleans to pack
%   packed_stream is the resulting booleans packed sequntially into uint64s
%
%   This works by reshaping the boolean array into a 32xN matrix and
%   multiplying that by the first 32 powers of 2. 32 is used because matrix
%   operations can only be done on doubles (not integers directly), and the
%   maximum int perfectly representable by double is less than the maximum
%   uint64, but greater than the maximum uint32. Basically this allows for
%   a quicker evaluation of bitshifting each of the bools into an int by
%   leveraging matlabs fast matrix double multiplication algorithms


% %%%%TODO->check how fast this is compared to the version below using the profiler
%     %check if the bitwidth is valid
%     bitwidths = [8 16 32 64];                       %regular bitwidths allowed
%     assert(any(width == bitwidths));                %verify that the specified bitwidth is allowed
%     dtypes = {@uint8, @uint16, @uint32, @uint32};   %functions for casting to differnt datatypes. uint64 is handled differently than the others, hence @uint32 instead
%     powers = [8 16 32 32];                          %maximum exponent for powers of 2 up to datatype width. uint64 case uses uint32 to prevent double overflow
%     
%     
%     power = powers(width==bitwidths);
%     dtype = dtypes{width==bitwidths};
%     
%     assert(mod(length(stream), width) == 0);   %to avoid overhead, of trimming array, make sure bits is divisible by the width 
%     num_ints = length(stream) / power;
%     pows = 2.^(0:power-1);
%     packed_stream = dtype(pows * reshape(double(stream), power, num_ints))';
%     
%     if width ~= 64
%         %convert the 32-bit array to a 64-bit array
%         packed_stream = typecast(packed_stream, 'uint64'); 
%     end

    
    % verify that the input data is valid 
    % i.e. valid width selected, and stream length is multiple of width
    assert(width == 8 || width == 16 || width == 32 || width == 64);
    assert(mod(length(stream), width) == 0);
    
    %first convert the boolean array into an array of bytes
    num_bytes = length(stream) / 8;
    pows = 2.^(0:7);
    packed_stream = uint8(pows * reshape(double(stream), 8, num_bytes))';
    
    %if desired width was not 8, typecast to the desired width
    if width == 16
        packed_stream = typecast(packed_stream, 'uint16');
    elseif width == 32
        packed_stream = typecast(packed_stream, 'uint32');
    elseif width == 64
        packed_stream = typecast(packed_stream, 'uint64');
    end
end



%this we have to pack with int32
%18446744073709551615 %max int64
%4294967295 %max uint32
%9007199254740992 %max representable int by double
