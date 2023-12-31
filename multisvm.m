function [itrfin] = multisvm(T, C, test)
    % Inputs: T=Training Matrix, C=Group, test=Testing matrix
    % Outputs: itrfin=Resultant class

    itrind = size(test, 1);
    itrfin = [];
    Cb = C;
    Tb = T;

    for tempind = 1:itrind
        tst = test(tempind, :);
        C = Cb;
        T = Tb;
        u = unique(C);
        N = length(u);
        c4 = [];
        c3 = [];
        j = 1;
        k = 1;

        if (N > 1)
            itr = 1;
            classes = 0;
            cond = max(C) - min(C);

            while ((classes ~= 1) && (itr <= length(u)) && (size(C, 2) > 1) && (cond > 0))
                % This while loop is the multiclass SVM Trick
                c1 = (C == u(itr));
                newClass = c1;

                % Train SVM using fitcsvm
                SVMModel = fitcsvm(T, newClass);

                % Classify using SVM
                classes = predict(SVMModel, tst);

                % This is the loop for Reduction of Training Set
                for i = 1:size(newClass, 2)
                    if newClass(1, i) == 0
                        c3(k, :) = T(i, :);
                        k = k + 1;
                    end
                end
                T = c3;
                c3 = [];
                k = 1;

                % This is the loop for reduction of group
                for i = 1:size(newClass, 2)
                    if newClass(1, i) == 0
                        c4(1, j) = C(1, i);
                        j = j + 1;
                    end
                end
                C = c4;
                c4 = [];
                j = 1;
                cond = max(C) - min(C);

                % This condition can select the particular value of iteration
                % based on classes
                if classes ~= 1
                    itr = itr + 1;
                end
            end
        end

        valt = Cb == u(itr);

        % This logic is used to allow classification of multiple rows testing matrix
        val = Cb(valt == 1);
        val = unique(val);
        itrfin(tempind, :) = val;
    end
end
