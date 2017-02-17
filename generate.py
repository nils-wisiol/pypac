from pypac import concepts, tools


n = 16
k = 4
N = 10

instance = concepts.IpMod2CombinedLtfs(n, k)
print('generated %s combined std Gaussian LTFs with %s inputs each, combined by the inner product mod 2 function' % (k, n))

print('printing %s samples' % N)
for x in tools.random_inputs(n, N):
    print('%s => %s' % (x, instance.eval(x)))
