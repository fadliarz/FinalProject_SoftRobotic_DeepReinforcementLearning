function g = ginv(g) % optimized on 31.05.2022
g = [ g(1,1), g(2,1), g(3,1), - g(1,1)*g(1,4) - g(2,1)*g(2,4) - g(3,1)*g(3,4);...
      g(1,2), g(2,2), g(3,2), - g(1,2)*g(1,4) - g(2,2)*g(2,4) - g(3,2)*g(3,4);...
      g(1,3), g(2,3), g(3,3), - g(1,3)*g(1,4) - g(2,3)*g(2,4) - g(3,3)*g(3,4);...
      0, 0, 0, 1];


