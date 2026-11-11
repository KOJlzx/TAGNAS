# Mean GPU utilisation over the training phase of an nvidia-smi `-l 1` log ("index, util" lines,
# one line per card per second). The phase starts at the first second from which the next 10 seconds
# average > 20% (all cards pooled): process start-up, data loading and tokenisation are excluded
# even if an early CUDA-context claim produced a stray busy sample before them.
{ sec = int((NR - 1) / ncards); u[sec] += $2; c[sec]++ ; if (sec > last) last = sec }
END {
  start = -1
  for (s = 0; s <= last - 9 && start < 0; s++) {
    t = 0; k = 0
    for (j = s; j < s + 10; j++) { t += u[j]; k += c[j] }
    if (k && t / k > 20) start = s
  }
  if (start < 0) { print "NA"; exit }
  t = 0; k = 0
  for (s = start; s <= last; s++) { t += u[s]; k += c[s] }
  printf "%.1f", t / k
}
