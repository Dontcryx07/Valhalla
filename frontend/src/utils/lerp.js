export function lerp(current, target, factor) {
  return current + (target - current) * factor;
}
