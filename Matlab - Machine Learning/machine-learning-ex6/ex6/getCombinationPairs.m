function [result] = getCombinationPairs(v1, v2)

[p, q] = meshgrid(v1, v2);

result = [p(:), q(:)];

end