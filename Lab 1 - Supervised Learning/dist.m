function dist = dist(x1,x2)
dist = 0;
for i = 1:length(x1)
    dist = dist + (x1(i) - x2(i))^2;
end
dist = sqrt(dist);
end
