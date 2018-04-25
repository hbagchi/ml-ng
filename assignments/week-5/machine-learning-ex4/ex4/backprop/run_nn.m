% prepare the XOR dataset
XOR = [0, 0, 0; 0, 1, 1; 1, 0, 1; 1, 1, 0];

%learn theta
[theta1, theta2] = xor_nn(XOR, 0.01, 10000);

% predict
for i = 1:rows(XOR)

    a1 = [1; XOR(i,1:2)'];
    z2 = theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = theta2 * a2;
    a3 = sigmoid(z3);
	
	disp('Hypothesis for '), disp(XOR(i, 1:2)), disp('is '), disp(a3);
end