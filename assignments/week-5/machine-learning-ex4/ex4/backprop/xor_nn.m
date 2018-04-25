% function to learn XOR solution using nn

function [theta1, theta2] = xor_nn(X, alpha, iterations)

  % initialze theta(s) randomly
  theta1 = 2 * rand(2, 3) - 1;
  theta2 = 2 * rand(1, 3) - 1;
  
  % delta accumulators for the partial derivatives
  theta1_delta = zeros(size(theta1));
  theta2_delta = zeros(size(theta2));
  
  % training set row size
  m = rows(X);

    for j = 1:iterations
	  for i = 1:m
		% do the forward propagation
		a1 = [1; X(i, 1:2)'];
		z2 = theta1 * a1;
		a2 = [1; sigmoid(z2)];
		z3 = theta2 * a2;
		a3 = sigmoid(z3);

		% calculate the error for the backprop
		delta3 = a3 - X(i, 3);
		delta2 = ((theta2' * delta3) .* (a2 .* (1 - a2)))(2:end);

		% add the deltas for this training example to the accumulators
		theta2_delta = theta2_delta + delta3 * a2';
		theta1_delta = theta1_delta + delta2 * a1';
	  end
      theta1 = theta1 - (alpha * (theta1_delta / m));
	  theta2 = theta2 - (alpha * (theta2_delta / m));
	  
	  if (mod(j, 2000) == 0)
		disp('Iterations Executed : '), disp(j);
	  end
	end
end
