function aspe = fair_stats(pred,y,s)
% Returns a vector of (accuracy,SP,EOP,EO)
    aspe = -ones(4,1);
    yv  = unique(y);
    if length(yv) == 2 && yv(1) == 0
        error("y should be {-1,1} in classification")
    end
    aspe(1) = mean(sign(pred) == y);
    aspe(2) = max(abs(corr(pred,s)));
    if length(yv) == 2
        aspe(3) = max(abs(corr(pred(y==1),s(y==1))));
        aspe(4) = max(aspe(3),max(abs(corr(pred(y~=1),s(y~=1)))));
    end
end
