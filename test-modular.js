// Quick test to verify Montgomery multiplication is working
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const addon = require('./index.node');

console.log('Available exports:', Object.keys(addon));
console.log('ModularArithmetic:', addon.ModularArithmetic);

try {
  console.log('\nTesting ModularArithmetic...');
  
  // Create a modular arithmetic engine with a small prime modulus
  const modulus = 17n;
  const mod = new addon.ModularArithmetic(Number(modulus));
  
  console.log('ModularArithmetic instance created');
  console.log('Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(mod)));
  
  console.log(`Modulus: ${mod.getModulus()}`);
  
  // Test modular addition
  const a = 10n;
  const b = 12n;
  const sum = mod.modAdd(Number(a), Number(b));
  console.log(`${a} + ${b} mod ${modulus} = ${sum} (expected ${(a + b) % modulus})`);
  
  // Test modular subtraction
  const diff = mod.modSub(Number(a), Number(b));
  const expected_diff = (a - b + modulus) % modulus;
  console.log(`${a} - ${b} mod ${modulus} = ${diff} (expected ${expected_diff})`);
  
  // Test Montgomery multiplication
  // First convert to Montgomery form
  const a_mont = mod.toMontgomery(Number(a));
  const b_mont = mod.toMontgomery(Number(b));
  console.log(`${a} in Montgomery form: ${a_mont}`);
  console.log(`${b} in Montgomery form: ${b_mont}`);
  
  // Multiply in Montgomery form
  const prod_mont = mod.montgomeryMul(a_mont, b_mont);
  console.log(`Montgomery multiplication result: ${prod_mont}`);
  
  // Convert back from Montgomery form
  const prod = mod.fromMontgomery(prod_mont);
  const expected_prod = (a * b) % modulus;
  console.log(`${a} * ${b} mod ${modulus} = ${prod} (expected ${expected_prod})`);
  
  if (prod === Number(expected_prod)) {
    console.log('\n✓ Montgomery multiplication is working correctly!');
  } else {
    console.log('\n✗ Montgomery multiplication failed!');
    process.exit(1);
  }
  
} catch (error) {
  console.error('Error:', error.message);
  console.error(error.stack);
  process.exit(1);
}
