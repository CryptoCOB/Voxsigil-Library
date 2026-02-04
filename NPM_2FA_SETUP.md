# 🔐 NPM 2FA Setup & Publishing

## Issue: 403 Forbidden - Two-Factor Authentication Required

npm now requires 2FA for publishing packages. Here's how to set it up:

---

## Option 1: Enable 2FA on Your Account (Recommended)

### Step 1: Login to npm Website
1. Go to https://www.npmjs.com/login
2. Login with your credentials

### Step 2: Enable 2FA
1. Click your profile icon (top right)
2. Go to **Account Settings**
3. Click **Two-Factor Authentication**
4. Choose **Enable 2FA**
5. Select **Authorization Only** (for publishing) or **Authorization and Writes** (more secure)
6. Scan QR code with authenticator app (Google Authenticator, Authy, etc.)
7. Enter the 6-digit code to verify

### Step 3: Publish with 2FA
```bash
npm publish --access public
# You'll be prompted for a one-time password (OTP)
# Enter the 6-digit code from your authenticator app
```

---

## Option 2: Use Access Token with 2FA Bypass (CI/CD)

### Step 1: Create Granular Access Token
1. Go to https://www.npmjs.com/settings/[your-username]/tokens
2. Click **Generate New Token**
3. Select **Granular Access Token**
4. Configure:
   - **Token Name**: `voxsigil-library-publish`
   - **Expiration**: 90 days (or custom)
   - **Packages and scopes**: Select `@voxsigil/*`
   - **Permissions**: 
     - ✅ Read and write packages
   - **Require 2FA**: ❌ Disable (allows bypass)

### Step 2: Use Token
```bash
# Set token in environment
export NPM_TOKEN=npm_xxxxxxxxxxxxxxxxxxxx

# Or create .npmrc file
echo "//registry.npmjs.org/:_authToken=\${NPM_TOKEN}" > ~/.npmrc

# Then publish
npm publish --access public
```

---

## Quick Fix for Right Now

If you want to publish immediately:

### 1. Enable 2FA (5 minutes)
1. Visit: https://www.npmjs.com/login
2. Account Settings → Two-Factor Authentication → Enable
3. Use your phone's authenticator app

### 2. Publish with OTP
```bash
npm publish --access public --otp=123456
# Replace 123456 with current code from authenticator
```

Or let npm prompt you:
```bash
npm publish --access public
# npm will prompt: "This operation requires a one-time password."
# Enter the 6-digit code from your authenticator app
```

---

## Testing After Publication

Once published successfully, verify:

```bash
# Check package exists
npm view @voxsigil/library version
# Should output: 2.0.0

# Test installation
mkdir /tmp/test-npm && cd /tmp/test-npm
npm install @voxsigil/library
node -e "const lib = require('@voxsigil/library'); console.log('✅ Works!');"
```

---

## Troubleshooting

### "Invalid OTP"
- Make sure your system clock is synchronized
- Generate a fresh code (codes expire after 30 seconds)
- Check you're using the correct authenticator app

### "403 Forbidden" Still Appears
- Clear npm cache: `npm cache clean --force`
- Re-login: `npm logout && npm login`
- Verify 2FA is enabled on npmjs.com

### "Package name too similar"
- npm may block if name conflicts with existing packages
- Choose a different package name if needed

---

## Security Best Practices

✅ **Do:**
- Use 2FA for your npm account
- Store access tokens securely (environment variables, secret managers)
- Use short-lived tokens (30-90 days)
- Revoke unused tokens

❌ **Don't:**
- Commit tokens to git
- Share tokens publicly
- Use the same token across multiple projects
- Disable 2FA without good reason

---

## Next Steps After Publishing

1. ✅ Verify package on npmjs.com
2. ✅ Test installation
3. ✅ Update MOLT_VISIBILITY_STATUS.md
4. ✅ Create GitHub release v2.0.0
5. ✅ Announce to community

---

**Current Status**: Package ready to publish, waiting for 2FA setup  
**Package Size**: 42.4 kB (23 files)  
**Version**: 2.0.0  
**Command**: `npm publish --access public` (after 2FA enabled)
